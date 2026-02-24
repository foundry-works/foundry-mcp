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
        # (skipped when the dedicated BRIEF phase already produced one)
        # ---------------------------------------------------------------
        if state.research_brief:
            logger.info(
                "Research brief already set by BRIEF phase (%d chars), "
                "skipping inline refinement",
                len(state.research_brief),
            )
        else:
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

        # ---------------------------------------------------------------
        # Step 3: Self-critique of sub-query decomposition
        # ---------------------------------------------------------------
        enable_critique = getattr(
            self.config, "deep_research_enable_planning_critique", True
        )
        if enable_critique and len(state.sub_queries) > 0:
            original_queries = [
                {"query": sq.query, "rationale": sq.rationale, "priority": sq.priority}
                for sq in state.sub_queries
            ]

            critique_prompt = self._build_decomposition_critique_prompt(state)
            self._check_cancellation(state)

            critique_result = await execute_llm_call(
                workflow=self,
                state=state,
                phase_name="planning_critique",
                system_prompt=self._build_critique_system_prompt(),
                user_prompt=critique_prompt,
                provider_id=None,  # Resolved by role
                model=None,  # Resolved by role
                temperature=0.2,  # Low temperature for analytical reasoning
                timeout=timeout,
                role="reflection",  # Uses cheap model
            )

            if isinstance(critique_result, WorkflowResult):
                # Critique failed — proceed without it (non-fatal)
                logger.warning(
                    "Planning critique LLM call failed, proceeding without critique: %s",
                    critique_result.error,
                )
                state.metadata["planning_critique"] = {
                    "original_sub_queries": original_queries,
                    "critique_response": None,
                    "adjusted_sub_queries": original_queries,
                    "error": critique_result.error or "LLM call failed",
                    "applied": False,
                }
            else:
                critique_content = critique_result.result.content or ""
                critique_parsed = self._parse_critique_response(critique_content)

                if critique_parsed["has_changes"]:
                    self._apply_critique_adjustments(state, critique_parsed)
                    logger.info(
                        "Planning critique applied: %d redundancies merged, "
                        "%d gaps added, %d adjustments",
                        len(critique_parsed.get("redundancies", [])),
                        len(critique_parsed.get("gaps", [])),
                        len(critique_parsed.get("adjustments", [])),
                    )
                else:
                    logger.info("Planning critique: no changes recommended")

                adjusted_queries = [
                    {"query": sq.query, "rationale": sq.rationale, "priority": sq.priority}
                    for sq in state.sub_queries
                ]
                state.metadata["planning_critique"] = {
                    "original_sub_queries": original_queries,
                    "critique_response": critique_content,
                    "critique_parsed": critique_parsed,
                    "adjusted_sub_queries": adjusted_queries,
                    "applied": critique_parsed["has_changes"],
                }

            # Persist updated state with critique
            self.memory.save_deep_research(state)

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

    # ------------------------------------------------------------------
    # Decomposition self-critique (Phase 2 of think-tool plan)
    # ------------------------------------------------------------------

    def _build_critique_system_prompt(self) -> str:
        """Build system prompt for the decomposition critique step.

        Returns:
            System prompt instructing analytical critique of sub-queries
        """
        return (
            "You are a research planning critic. Your task is to evaluate a set "
            "of sub-queries generated for a research project and identify issues "
            "with the decomposition quality.\n\n"
            "You evaluate for:\n"
            "1. Redundancies — sub-queries that overlap significantly\n"
            "2. Missing perspectives — important angles not covered\n"
            "3. Scope issues — queries that are too broad or too narrow\n\n"
            "Your response MUST be valid JSON with this exact structure:\n"
            "{\n"
            '    "redundancies": [\n'
            '        {"indices": [0, 2], "reason": "Both cover AI safety regulations", '
            '"merged_query": "AI safety regulations and governance policies"}\n'
            "    ],\n"
            '    "gaps": [\n'
            '        {"query": "A new sub-query to add", "rationale": "Covers missing perspective", "priority": 1}\n'
            "    ],\n"
            '    "adjustments": [\n'
            '        {"index": 1, "revised_query": "More focused version", "reason": "Original too broad"}\n'
            "    ],\n"
            '    "assessment": "Brief overall assessment of decomposition quality"\n'
            "}\n\n"
            "Guidelines:\n"
            "- Only flag TRUE redundancies (significant content overlap, not just related topics)\n"
            "- Only add gaps for GENUINELY missing critical perspectives\n"
            "- Keep the total sub-query count reasonable (2-7 after changes)\n"
            "- If the decomposition is already good, return empty arrays\n"
            "- merged_query in redundancies replaces ALL the redundant queries with one\n\n"
            "IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."
        )

    def _build_decomposition_critique_prompt(self, state: DeepResearchState) -> str:
        """Build critique prompt from generated sub-queries and research brief.

        Presents the sub-queries back to an LLM for evaluation of redundancies,
        missing perspectives, and scope issues.

        Args:
            state: Current research state with sub_queries populated

        Returns:
            Critique prompt string
        """
        parts: list[str] = [
            f"# Research Brief\n{state.research_brief or state.original_query}\n",
            "# Generated Sub-Queries\n",
        ]

        for i, sq in enumerate(state.sub_queries):
            parts.append(
                f"{i}. **{sq.query}**\n"
                f"   Rationale: {sq.rationale or 'N/A'}\n"
                f"   Priority: {sq.priority}"
            )

        parts.extend([
            "",
            "# Instructions",
            "Evaluate the sub-queries above for:",
            "1. **Redundancies**: Are any sub-queries covering essentially the same ground?",
            "2. **Missing perspectives**: Are there important angles missing? "
            "Consider: historical context, economic factors, technical details, "
            "stakeholder perspectives, comparative analysis, recent developments.",
            "3. **Scope issues**: Are any sub-queries too broad (need splitting) or "
            "too narrow (could be merged with another)?",
            "",
            "Return your critique as JSON.",
        ])

        return "\n".join(parts)

    def _parse_critique_response(self, content: str) -> dict[str, Any]:
        """Parse the critique LLM response into structured adjustments.

        Args:
            content: Raw LLM response content

        Returns:
            Dict with 'redundancies', 'gaps', 'adjustments', 'assessment',
            and 'has_changes' keys
        """
        result: dict[str, Any] = {
            "redundancies": [],
            "gaps": [],
            "adjustments": [],
            "assessment": "",
            "has_changes": False,
        }

        if not content:
            return result

        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in planning critique response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from planning critique: %s", e)
            return result

        # Parse redundancies
        raw_redundancies = data.get("redundancies", [])
        if isinstance(raw_redundancies, list):
            for r in raw_redundancies:
                if not isinstance(r, dict):
                    continue
                indices = r.get("indices", [])
                merged = r.get("merged_query", "").strip()
                if isinstance(indices, list) and len(indices) >= 2 and merged:
                    # Validate indices are integers
                    try:
                        int_indices = [int(i) for i in indices]
                    except (ValueError, TypeError):
                        continue
                    result["redundancies"].append({
                        "indices": int_indices,
                        "reason": r.get("reason", ""),
                        "merged_query": merged,
                    })

        # Parse gaps (new queries to add)
        raw_gaps = data.get("gaps", [])
        if isinstance(raw_gaps, list):
            for g in raw_gaps:
                if not isinstance(g, dict):
                    continue
                query = g.get("query", "").strip()
                if query:
                    result["gaps"].append({
                        "query": query,
                        "rationale": g.get("rationale", ""),
                        "priority": min(max(int(g.get("priority", 1)), 1), 10),
                    })

        # Parse scope adjustments
        raw_adjustments = data.get("adjustments", [])
        if isinstance(raw_adjustments, list):
            for a in raw_adjustments:
                if not isinstance(a, dict):
                    continue
                revised = a.get("revised_query", "").strip()
                if revised:
                    try:
                        idx = int(a.get("index", -1))
                    except (ValueError, TypeError):
                        continue
                    result["adjustments"].append({
                        "index": idx,
                        "revised_query": revised,
                        "reason": a.get("reason", ""),
                    })

        result["assessment"] = data.get("assessment", "")
        result["has_changes"] = bool(
            result["redundancies"] or result["gaps"] or result["adjustments"]
        )

        return result

    def _apply_critique_adjustments(
        self,
        state: DeepResearchState,
        critique: dict[str, Any],
    ) -> None:
        """Apply parsed critique adjustments to state.sub_queries.

        Processes in order:
        1. Scope adjustments (revise query text in-place)
        2. Redundancy merges (replace redundant queries with merged version)
        3. Gap additions (add new sub-queries for missing perspectives)

        Respects ``state.max_sub_queries`` bound after all changes.

        Args:
            state: Current research state (sub_queries modified in-place)
            critique: Parsed critique from _parse_critique_response
        """
        current_queries = list(state.sub_queries)

        # 1. Apply scope adjustments (revise query text)
        for adj in critique.get("adjustments", []):
            idx = adj["index"]
            if 0 <= idx < len(current_queries):
                old_query = current_queries[idx].query
                current_queries[idx].query = adj["revised_query"]
                logger.debug(
                    "Critique adjustment [%d]: '%s' -> '%s'",
                    idx,
                    old_query[:60],
                    adj["revised_query"][:60],
                )

        # 2. Merge redundancies (collect indices to remove, add merged query)
        indices_to_remove: set[int] = set()
        merged_queries: list[dict[str, Any]] = []
        for red in critique.get("redundancies", []):
            valid_indices = [i for i in red["indices"] if 0 <= i < len(current_queries)]
            if len(valid_indices) < 2:
                continue
            # Keep the best (lowest number) priority from the merged set
            min_priority = min(
                current_queries[i].priority for i in valid_indices
            )
            indices_to_remove.update(valid_indices)
            merged_queries.append({
                "query": red["merged_query"],
                "rationale": red.get("reason", "Merged from redundant sub-queries"),
                "priority": min_priority,
            })

        # Remove redundant queries
        if indices_to_remove:
            current_queries = [
                sq for i, sq in enumerate(current_queries)
                if i not in indices_to_remove
            ]

        # Replace state.sub_queries with the filtered list BEFORE adding new ones
        state.sub_queries = current_queries

        # Add merged queries via add_sub_query (generates proper IDs)
        for mq in merged_queries:
            state.add_sub_query(
                query=mq["query"],
                rationale=mq["rationale"],
                priority=mq["priority"],
            )

        # 3. Add gap queries (missing perspectives)
        for gap in critique.get("gaps", []):
            if len(state.sub_queries) >= state.max_sub_queries:
                logger.debug("Critique gap skipped (at max_sub_queries limit): %s", gap["query"][:60])
                break
            state.add_sub_query(
                query=gap["query"],
                rationale=gap.get("rationale", "Added from planning critique"),
                priority=gap.get("priority", 1),
            )

        # Final safety: ensure we don't exceed max_sub_queries
        if len(state.sub_queries) > state.max_sub_queries:
            # Keep highest-priority queries (lowest priority number)
            state.sub_queries.sort(key=lambda sq: sq.priority)
            state.sub_queries = state.sub_queries[: state.max_sub_queries]
