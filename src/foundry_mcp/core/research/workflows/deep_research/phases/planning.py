"""Planning phase mixin for DeepResearchWorkflow.

Decomposes the original research query into focused sub-queries via LLM call.
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


class PlanningPhaseMixin:
    """Planning phase methods. Mixed into DeepResearchWorkflow.

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

    async def _execute_planning_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute planning phase: decompose query into sub-queries.

        This phase:
        1. Analyzes the original research query
        2. Generates a research brief explaining the approach
        3. Decomposes the query into 2-5 focused sub-queries
        4. Assigns priorities to each sub-query

        Args:
            state: Current research state
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with planning outcome
        """
        logger.info("Starting planning phase for query: %s", state.original_query[:100])

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "planning",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # ---------------------------------------------------------------
        # Step 1: Refine the raw query into a structured research brief
        # ---------------------------------------------------------------
        brief_prompt = self._build_brief_refinement_prompt(state)
        self._check_cancellation(state)

        brief_call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="planning",
            system_prompt=(
                "You are a research brief writer. Rewrite the user's research "
                "request into a single, precise research brief paragraph."
            ),
            user_prompt=brief_prompt,
            provider_id=provider_id or state.planning_provider,
            model=state.planning_model,
            temperature=0.3,  # Low creativity — faithful rewrite
            timeout=timeout,
            role="summarization",  # Cheap model for lightweight rewrite
        )

        if isinstance(brief_call_result, WorkflowResult):
            # Brief refinement failed — fall back to using the raw query
            logger.warning(
                "Brief refinement LLM call failed, using raw query as brief"
            )
            state.research_brief = state.original_query
        else:
            refined_brief = (brief_call_result.result.content or "").strip()
            state.research_brief = refined_brief or state.original_query
            logger.info(
                "Brief refinement complete (%d chars)",
                len(state.research_brief),
            )

        # ---------------------------------------------------------------
        # Step 2: Decompose the (refined) brief into sub-queries
        # ---------------------------------------------------------------
        system_prompt = self._build_planning_system_prompt(state)
        user_prompt = self._build_planning_user_prompt(state)

        self._check_cancellation(state)

        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="planning",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.planning_provider,
            model=state.planning_model,
            temperature=0.7,  # Some creativity for diverse sub-queries
            timeout=timeout,
            role="research",
        )
        if isinstance(call_result, WorkflowResult):
            return call_result  # Error path
        result = call_result.result

        # Parse the response
        parsed = self._parse_planning_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse planning response, using fallback")
            # Fallback: treat entire query as single sub-query.
            # Only overwrite research_brief if it wasn't already set by
            # the brief-refinement step.
            if not state.research_brief:
                state.research_brief = f"Direct research on: {state.original_query}"
            state.add_sub_query(
                query=state.original_query,
                rationale="Original query used directly due to parsing failure",
                priority=1,
            )
        else:
            # Keep the refined brief from step 1; only fall back to the
            # planning response's brief if refinement didn't produce one.
            if not state.research_brief:
                state.research_brief = parsed["research_brief"]
            for sq in parsed["sub_queries"]:
                state.add_sub_query(
                    query=sq["query"],
                    rationale=sq.get("rationale"),
                    priority=sq.get("priority", 1),
                )

        # Save state after planning
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "planning_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "parse_success": parsed["success"],
                "research_brief": state.research_brief,
                "sub_queries": [
                    {
                        "id": sq.id,
                        "query": sq.query,
                        "rationale": sq.rationale,
                        "priority": sq.priority,
                    }
                    for sq in state.sub_queries
                ],
            },
        )

        logger.info(
            "Planning phase complete: %d sub-queries generated",
            len(state.sub_queries),
        )

        finalize_phase(self, state, "planning", phase_start_time)

        return WorkflowResult(
            success=True,
            content=state.research_brief or "Planning complete",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "sub_query_count": len(state.sub_queries),
                "research_brief": state.research_brief,
            },
        )

    def _build_planning_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for query decomposition.

        Args:
            state: Current research state (reserved for future state-aware prompts)

        Returns:
            System prompt string
        """
        # state is reserved for future state-aware prompt customization
        _ = state
        return """You are a research planning assistant. Your task is to analyze a research query and decompose it into focused sub-queries that can be researched independently.

Your response MUST be valid JSON with this exact structure:
{
    "research_brief": "A 2-3 sentence summary of the research approach and what aspects will be investigated",
    "sub_queries": [
        {
            "query": "A specific, focused search query",
            "rationale": "Why this sub-query is important for the research",
            "priority": 1
        }
    ]
}

Guidelines:
- Generate 2-5 sub-queries (aim for 3-4 typically)
- Each sub-query should focus on a distinct aspect of the research
- Queries should be specific enough to yield relevant search results
- Priority 1 is highest (most important), higher numbers are lower priority
- Avoid overlapping queries - each should cover unique ground
- Consider different angles: definition, examples, comparisons, recent developments, expert opinions

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_brief_refinement_prompt(self, state: DeepResearchState) -> str:
        """Build a prompt that refines a raw user query into a focused research brief.

        The brief-refinement step transforms potentially vague or underspecified
        user queries into a single, focused research paragraph that maximizes
        specificity, leaves genuinely unstated dimensions open-ended (rather than
        assuming), and biases toward primary/official sources.

        This runs *before* sub-query decomposition so the planner operates on a
        well-structured brief rather than the raw user input.

        Args:
            state: Current research state containing original_query and optional
                   system_prompt / clarification_constraints.

        Returns:
            A user prompt string for the brief-refinement LLM call.
        """
        parts: list[str] = [
            "Transform the following user research request into a single, focused "
            "research brief paragraph.\n",
            "Rules:",
            "1. Maximize specificity: extract every concrete detail the user provided "
            "(names, dates, versions, quantities) and foreground them.",
            "2. Do NOT assume values for dimensions the user left unspecified. Instead, "
            "note them as open questions the research should address.",
            "3. Bias toward primary and official sources (specifications, documentation, "
            "peer-reviewed work, government datasets) over aggregators or secondary commentary.",
            "4. Preserve the user's language — write the brief in the same language "
            "as the query below.",
            "5. Output ONLY the research brief paragraph, nothing else.\n",
            f"User query:\n{state.original_query}",
        ]

        if state.system_prompt:
            parts.append(f"\nAdditional context provided by the user:\n{state.system_prompt}")

        if state.clarification_constraints:
            parts.append("\nClarification constraints (already confirmed with the user):")
            for key, value in state.clarification_constraints.items():
                parts.append(f"- {key}: {value}")

        return "\n".join(parts)

    def _build_planning_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt for query decomposition.

        Uses the refined research brief (``state.research_brief``) when
        available so sub-query decomposition operates on a more specific,
        structured input rather than the raw user query.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        # Prefer the refined brief; fall back to original query if unset.
        query_input = state.research_brief or state.original_query

        prompt = f"""Research Query: {query_input}

Please decompose this research query into {state.max_sub_queries} or fewer focused sub-queries.

Consider:
1. What are the key aspects that need investigation?
2. What background information would help understand this topic?
3. What specific questions would lead to comprehensive coverage?
4. What different perspectives or sources might be valuable?

Generate the research plan as JSON."""

        # Add custom system prompt context if provided
        if state.system_prompt:
            prompt += f"\n\nAdditional context: {state.system_prompt}"

        # Add clarification constraints if available (from clarification phase)
        if state.clarification_constraints:
            prompt += "\n\nClarification constraints (use these to focus the research):"
            for key, value in state.clarification_constraints.items():
                prompt += f"\n- {key}: {value}"

        return prompt

    def _parse_planning_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured planning data.

        Attempts to extract JSON from the response, with fallback handling
        for various response formats.

        Args:
            content: Raw LLM response content
            state: Current research state (for max_sub_queries limit)

        Returns:
            Dict with 'success', 'research_brief', and 'sub_queries' keys
        """
        result: dict[str, Any] = {
            "success": False,
            "research_brief": None,
            "sub_queries": [],
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in planning response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from planning response: %s", e)
            return result

        # Extract research brief
        result["research_brief"] = data.get("research_brief", "")

        # Extract and validate sub-queries
        raw_queries = data.get("sub_queries", [])
        if not isinstance(raw_queries, list):
            logger.warning("sub_queries is not a list")
            return result

        for i, sq in enumerate(raw_queries):
            if not isinstance(sq, dict):
                continue
            query = sq.get("query", "").strip()
            if not query:
                continue

            # Limit to max_sub_queries
            if len(result["sub_queries"]) >= state.max_sub_queries:
                break

            result["sub_queries"].append(
                {
                    "query": query,
                    "rationale": sq.get("rationale", ""),
                    "priority": min(max(int(sq.get("priority", i + 1)), 1), 10),
                }
            )

        # Mark success if we got at least one sub-query
        result["success"] = len(result["sub_queries"]) > 0

        return result
