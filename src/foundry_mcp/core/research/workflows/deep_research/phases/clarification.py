"""Clarification phase mixin for DeepResearchWorkflow.

Analyzes query specificity and optionally generates clarifying questions
before the planning phase begins. When enabled, this reduces wasted search
credits on vague or ambiguous queries by inferring constraints upfront.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    extract_json,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)


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
        """Execute clarification phase: assess query specificity and infer constraints.

        This phase:
        1. Sends the original query to a fast model for specificity assessment
        2. If the query is specific enough, proceeds immediately (no-op)
        3. If vague, infers reasonable constraints (scope, timeframe, domain)
           and stores them in state.clarification_constraints
        4. The inferred constraints are fed into the planning phase

        Since this runs non-interactively (MCP tool response is returned after
        the full workflow), we infer constraints rather than asking the user
        and blocking. The clarification questions are recorded in state metadata
        for transparency.

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

        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="clarification",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id,
            model=None,
            temperature=0.3,  # Low temperature for analytical assessment
            timeout=timeout,
        )
        if isinstance(call_result, WorkflowResult):
            return call_result  # Error path
        result = call_result.result

        parsed = self._parse_clarification_response(result.content)

        if parsed["needs_clarification"]:
            state.clarification_constraints = parsed.get("inferred_constraints", {})
            state.metadata["clarification_questions"] = parsed.get("questions", [])
            logger.info(
                "Clarification phase: query needs refinement, inferred %d constraints",
                len(state.clarification_constraints),
            )
        else:
            logger.info("Clarification phase: query is specific enough, no constraints needed")

        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "clarification_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "needs_clarification": parsed["needs_clarification"],
                "questions": parsed.get("questions", []),
                "inferred_constraints": parsed.get("inferred_constraints", {}),
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
                "needs_clarification": parsed["needs_clarification"],
                "constraints_count": len(state.clarification_constraints),
            },
        )

    def _build_clarification_system_prompt(self) -> str:
        """Build system prompt for query clarification assessment.

        Returns:
            System prompt string
        """
        return """You are a research query analyst. Your task is to evaluate whether a research query is specific enough for focused, high-quality research.

Analyze the query and respond with valid JSON in this exact structure:
{
    "needs_clarification": true/false,
    "questions": [
        "Clarifying question 1?",
        "Clarifying question 2?"
    ],
    "inferred_constraints": {
        "scope": "description of inferred scope",
        "timeframe": "description of inferred timeframe (if relevant)",
        "domain": "specific domain or field to focus on",
        "depth": "overview | detailed | comprehensive",
        "geographic_focus": "region or 'global' (if relevant)"
    }
}

Rules:
- Set "needs_clarification" to true if the query is vague, overly broad, or ambiguous
- Set "needs_clarification" to false if the query is already specific and actionable
- Generate 1-3 clarifying questions that would most improve research focus
- ALWAYS provide "inferred_constraints" with your best inference of what the user likely wants
- Only include constraint keys that are relevant (omit irrelevant ones)
- The constraints should narrow the research to produce focused, useful results
- Be practical: infer the most likely intent rather than asking about edge cases

Examples of vague queries needing clarification:
- "How does AI work?" → Too broad, needs scope (ML? generative AI? robotics?)
- "What's the best database?" → Missing context (use case, scale, budget)
- "Tell me about climate change" → Needs focus (causes? solutions? policy? economics?)

Examples of specific queries NOT needing clarification:
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
        """Parse LLM response into structured clarification data.

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

        result["needs_clarification"] = bool(data.get("needs_clarification", False))

        questions = data.get("questions", [])
        if isinstance(questions, list):
            result["questions"] = [str(q) for q in questions[:3] if q]

        constraints = data.get("inferred_constraints", {})
        if isinstance(constraints, dict):
            # Only keep string-valued constraints, filter empty values
            result["inferred_constraints"] = {
                k: str(v) for k, v in constraints.items() if v and isinstance(v, (str, int, float, bool))
            }

        return result
