"""Clarification phase mixin for DeepResearchWorkflow.

Analyzes query specificity and generates a structured binary decision:
either a clarifying question (need_clarification=True) or a verification
statement confirming the LLM's understanding (need_clarification=False).

When clarification is not needed, the verification text is stored in
state.clarification_constraints for traceability and fed into the
planning phase.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    ClarificationDecision,
    extract_json,
    parse_clarification_decision,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_structured_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)


def _strict_parse_clarification(content: str) -> ClarificationDecision:
    """Parse clarification response with strict validation.

    Raises on failure so that ``execute_structured_llm_call`` can retry.

    Args:
        content: Raw LLM response content

    Returns:
        ClarificationDecision with extracted fields

    Raises:
        ValueError: If no valid JSON found or required field missing
        json.JSONDecodeError: If JSON is malformed
    """
    json_str = extract_json(content)
    if not json_str:
        raise ValueError("No JSON object found in clarification response")
    data = json.loads(json_str)
    if "need_clarification" not in data:
        raise ValueError("Missing required 'need_clarification' field in response")
    return ClarificationDecision(
        need_clarification=bool(data["need_clarification"]),
        question=str(data.get("question", "")),
        verification=str(data.get("verification", "")),
    )


class ClarificationPhaseMixin:
    """Clarification phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...

    async def _execute_clarification_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute clarification phase with structured binary decision.

        Uses ``execute_structured_llm_call`` to obtain a JSON response
        with the schema ``{need_clarification, question, verification}``.

        Behavior:
        - ``need_clarification=True``: stores the question in state metadata
          and infers constraints (existing flow).
        - ``need_clarification=False``: stores the verification text in
          ``state.clarification_constraints`` for traceability and logs an
          audit event, then proceeds to planning.
        - Parse failure (after 3 retries): treats as "no clarification
          needed" — safe default matching existing behavior.

        Args:
            state: Current research state
            provider_id: LLM provider to use (preferably a fast model)
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with clarification outcome
        """
        logger.info("Starting clarification phase for query: %s", state.original_query[:100])

        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "clarification",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        system_prompt = self._build_clarification_system_prompt()
        user_prompt = self._build_clarification_user_prompt(state)

        self._check_cancellation(state)

        call_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="clarification",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id,
            model=None,
            temperature=0.3,  # Low temperature for analytical assessment
            timeout=timeout,
            parse_fn=_strict_parse_clarification,
            role="clarification",
        )

        # LLM-level error — propagate immediately
        if isinstance(call_result, WorkflowResult):
            return call_result

        result = call_result.result

        # Determine the clarification decision
        if call_result.parsed is not None:
            decision: ClarificationDecision = call_result.parsed
        else:
            # Parse exhausted — use lenient fallback on the last response
            decision = parse_clarification_decision(result.content or "")

        if decision.need_clarification:
            # Store question for transparency; infer constraints from existing
            # parsing for backward compatibility
            state.metadata["clarification_questions"] = (
                [decision.question] if decision.question else []
            )
            # Attempt to extract inferred_constraints from the raw response
            # for backward compatibility with planning phase
            legacy_parsed = self._parse_clarification_response(result.content)
            state.clarification_constraints = legacy_parsed.get("inferred_constraints", {})
            logger.info(
                "Clarification phase: query needs refinement, question=%s",
                decision.question[:100] if decision.question else "(none)",
            )
        else:
            # Store verification as a constraint for traceability
            if decision.verification:
                state.clarification_constraints = {
                    "verification": decision.verification,
                }
            logger.info(
                "Clarification phase: query understood, verification=%s",
                decision.verification[:100] if decision.verification else "(none)",
            )
            self._write_audit_event(
                state,
                "clarification_verification",
                data={
                    "verification": decision.verification,
                    "task_id": state.id,
                },
            )

        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "clarification_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "need_clarification": decision.need_clarification,
                "question": decision.question,
                "verification": decision.verification,
                "parse_retries": call_result.parse_retries,
            },
        )

        finalize_phase(self, state, "clarification", phase_start_time)

        return WorkflowResult(
            success=True,
            content="Clarification complete",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "need_clarification": decision.need_clarification,
                "constraints_count": len(state.clarification_constraints),
                "parse_retries": call_result.parse_retries,
            },
        )

    def _build_clarification_system_prompt(self) -> str:
        """Build system prompt for structured clarification decision.

        Returns:
            System prompt string requesting JSON with the
            ``{need_clarification, question, verification}`` schema.
        """
        return """You are a research query analyst. Your task is to evaluate whether a research query is specific enough for focused, high-quality research.

Make a binary decision and respond with valid JSON in this exact structure:
{
    "need_clarification": true/false,
    "question": "A single clarifying question if clarification is needed, otherwise empty string",
    "verification": "Your restatement of how you understand the query and what you will research"
}

Rules:
- Set "need_clarification" to true ONLY if the query is genuinely vague, overly broad, or ambiguous
- Set "need_clarification" to false if the query is specific enough for focused research
- When "need_clarification" is true: provide the single most important clarifying question in "question"
- When "need_clarification" is false: provide your understanding of the query in "verification" — restate what you believe the user wants researched, including scope, domain, and focus
- ALWAYS provide "verification" regardless of the decision — it documents your understanding

Examples of vague queries needing clarification (need_clarification=true):
- "How does AI work?" → Too broad, needs scope (ML? generative AI? robotics?)
- "What's the best database?" → Missing context (use case, scale, budget)
- "Tell me about climate change" → Needs focus (causes? solutions? policy? economics?)

Examples of specific queries NOT needing clarification (need_clarification=false):
- "Compare PostgreSQL vs MySQL for high-write OLTP workloads in 2024"
- "What are the current FDA regulations for AI-based medical devices?"
- "How does the Rust borrow checker prevent data races?"

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_clarification_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt with the original query for assessment.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        prompt = f"Research Query: {state.original_query}"

        if state.system_prompt:
            prompt += f"\n\nAdditional context provided by user: {state.system_prompt}"

        return prompt

    def _parse_clarification_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into structured clarification data (legacy).

        Retained for backward compatibility — extracts inferred_constraints
        from responses that may use the old schema format.

        Args:
            content: Raw LLM response content

        Returns:
            Dict with 'needs_clarification', 'questions', and 'inferred_constraints'
        """
        result: dict[str, Any] = {
            "needs_clarification": False,
            "questions": [],
            "inferred_constraints": {},
        }

        if not content:
            return result

        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in clarification response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from clarification response: %s", e)
            return result

        result["needs_clarification"] = bool(
            data.get("needs_clarification", data.get("need_clarification", False))
        )

        questions = data.get("questions", [])
        if isinstance(questions, list):
            result["questions"] = [str(q) for q in questions[:3] if q]

        constraints = data.get("inferred_constraints", {})
        if isinstance(constraints, dict):
            # Only keep string-valued constraints, filter empty values
            result["inferred_constraints"] = {
                k: (str(v).lower() if isinstance(v, bool) else str(v))
                for k, v in constraints.items()
                if v is not None and v != "" and isinstance(v, (str, int, float, bool))
            }

        return result
