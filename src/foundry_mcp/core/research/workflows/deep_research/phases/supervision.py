"""Supervision phase mixin for DeepResearchWorkflow.

Assesses coverage of completed sub-queries and generates follow-up queries
to fill gaps before proceeding to analysis. Modeled after the iterative
supervisor loop in open_deep_research, adapted for foundry-mcp's
single-prompt architecture.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

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

# Maximum follow-up queries the supervisor can generate per round
_MAX_FOLLOW_UPS_PER_ROUND = 3


class SupervisionPhaseMixin:
    """Supervision phase methods. Mixed into DeepResearchWorkflow.

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

    async def _execute_supervision_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute supervision phase: assess coverage and generate follow-up queries.

        This phase:
        1. Builds per-sub-query coverage data (source count, quality, domains)
        2. If heuristic says all queries are sufficiently covered AND round > 0,
           skips LLM call and returns should_continue_gathering=False
        3. Calls LLM for coverage assessment with structured JSON output
        4. Parses response, deduplicates follow-up queries
        5. Adds new sub-queries (priority 2) with budget cap
        6. Increments supervision_round and records history

        Args:
            state: Current research state with completed sub-queries
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with metadata["should_continue_gathering"] flag
        """
        min_sources = getattr(
            self.config,
            "deep_research_supervision_min_sources_per_query",
            2,
        )

        # Build coverage data for all sub-queries
        coverage_data = self._build_per_query_coverage(state)

        # Heuristic early-exit: if all queries are sufficiently covered
        # and we've done at least one round, skip the LLM call
        if state.supervision_round > 0:
            heuristic = self._assess_coverage_heuristic(state, min_sources)
            if not heuristic["should_continue_gathering"]:
                logger.info(
                    "Supervision heuristic: all queries sufficiently covered "
                    "(round %d), advancing to analysis",
                    state.supervision_round,
                )
                self._write_audit_event(
                    state,
                    "supervision_result",
                    data={
                        "reason": "heuristic_sufficient",
                        "supervision_round": state.supervision_round,
                        "coverage_summary": heuristic,
                    },
                )
                # Record in supervision history
                history = state.metadata.setdefault("supervision_history", [])
                history.append(
                    {
                        "round": state.supervision_round,
                        "method": "heuristic",
                        "should_continue_gathering": False,
                        "follow_ups_added": 0,
                        "overall_coverage": "sufficient",
                    }
                )
                state.supervision_round += 1
                self.memory.save_deep_research(state)
                return WorkflowResult(
                    success=True,
                    content="Coverage sufficient by heuristic, advancing to analysis",
                    metadata={
                        "research_id": state.id,
                        "iteration": state.iteration,
                        "supervision_round": state.supervision_round,
                        "should_continue_gathering": False,
                        "method": "heuristic",
                    },
                )

        logger.info(
            "Starting supervision phase: round %d/%d, %d completed sub-queries, %d sources",
            state.supervision_round,
            state.max_supervision_rounds,
            len(state.completed_sub_queries()),
            len(state.sources),
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "supervision",
                "iteration": state.iteration,
                "supervision_round": state.supervision_round,
                "task_id": state.id,
            },
        )

        # --- Think step: deliberate on gaps before generating follow-ups ---
        # The think step is a separate LLM call using a cheap model that
        # forces explicit gap analysis. Its output is fed into the follow-up
        # generation prompt so queries are grounded in deliberate reasoning.
        # Skip on round > 0 (the first round already provides grounding).
        think_output: Optional[str] = None
        if state.supervision_round == 0:
            think_prompt = self._build_think_prompt(state, coverage_data)
            think_system = self._build_think_system_prompt()

            self._check_cancellation(state)

            think_result = await execute_llm_call(
                workflow=self,
                state=state,
                phase_name="supervision_think",
                system_prompt=think_system,
                user_prompt=think_prompt,
                provider_id=None,  # Resolved by role
                model=None,  # Resolved by role
                temperature=0.2,  # Low temperature for analytical reasoning
                timeout=getattr(self.config, "deep_research_reflection_timeout", 60.0),
                role="reflection",  # Uses cheap model
            )
            if isinstance(think_result, WorkflowResult):
                # Think step failed — proceed without it (non-fatal)
                logger.warning(
                    "Supervision think step failed, proceeding without gap analysis: %s",
                    think_result.error,
                )
            else:
                think_output = think_result.result.content
                logger.info(
                    "Supervision think step completed: %d chars, %s tokens",
                    len(think_output or ""),
                    think_result.result.tokens_used,
                )

        # Build prompts (with think output if available)
        system_prompt = self._build_supervision_system_prompt(state)
        user_prompt = self._build_supervision_user_prompt(
            state, coverage_data, think_output
        )

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # Execute LLM call with lifecycle instrumentation
        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.supervision_provider,
            model=state.supervision_model,
            temperature=0.3,
            timeout=timeout,
            role="supervision",
        )
        if isinstance(call_result, WorkflowResult):
            # LLM call failed — fall back to heuristic
            logger.warning(
                "Supervision LLM call failed, falling back to heuristic assessment"
            )
            heuristic = self._assess_coverage_heuristic(state, min_sources)
            history = state.metadata.setdefault("supervision_history", [])
            history.append(
                {
                    "round": state.supervision_round,
                    "method": "heuristic_fallback",
                    "should_continue_gathering": heuristic["should_continue_gathering"],
                    "follow_ups_added": 0,
                    "overall_coverage": heuristic.get("overall_coverage", "unknown"),
                    "error": call_result.error or "LLM call failed",
                }
            )
            state.supervision_round += 1
            self.memory.save_deep_research(state)
            return WorkflowResult(
                success=True,
                content="Supervision fell back to heuristic assessment",
                metadata={
                    "research_id": state.id,
                    "iteration": state.iteration,
                    "supervision_round": state.supervision_round,
                    "should_continue_gathering": heuristic["should_continue_gathering"],
                    "method": "heuristic_fallback",
                },
            )
        result = call_result.result

        # Parse the LLM response
        parsed = self._parse_supervision_response(result.content, state)

        # Determine how many new sub-queries we can add (budget cap)
        max_sub_queries = getattr(state, "max_sub_queries", 10)
        budget_remaining = max(0, max_sub_queries - len(state.sub_queries))

        # Add follow-up queries as new pending sub-queries
        follow_ups = parsed.get("follow_up_queries", [])
        capped_follow_ups = follow_ups[:min(budget_remaining, _MAX_FOLLOW_UPS_PER_ROUND)]
        new_sub_queries = 0
        for fq in capped_follow_ups:
            state.add_sub_query(
                query=fq["query"],
                rationale=fq.get("rationale", "Follow-up from supervision"),
                priority=fq.get("priority", 2),
            )
            new_sub_queries += 1

        # Determine should_continue_gathering
        should_continue = parsed.get("should_continue_gathering", False) and new_sub_queries > 0

        # Increment supervision round
        state.supervision_round += 1

        # Record supervision history (including think output for traceability)
        history = state.metadata.setdefault("supervision_history", [])
        history_entry: dict[str, Any] = {
            "round": state.supervision_round,
            "method": "llm",
            "should_continue_gathering": should_continue,
            "follow_ups_added": new_sub_queries,
            "overall_coverage": parsed.get("overall_coverage", "unknown"),
            "per_query_assessment": parsed.get("per_query_assessment", []),
            "rationale": parsed.get("rationale", ""),
        }
        if think_output:
            history_entry["think_output"] = think_output
        history.append(history_entry)

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "supervision_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "supervision_round": state.supervision_round,
                "follow_ups_added": new_sub_queries,
                "should_continue_gathering": should_continue,
                "overall_coverage": parsed.get("overall_coverage", "unknown"),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
            },
        )

        logger.info(
            "Supervision round %d complete: %d follow-up queries, continue_gathering=%s",
            state.supervision_round,
            new_sub_queries,
            should_continue,
        )

        finalize_phase(self, state, "supervision", phase_start_time)

        return WorkflowResult(
            success=True,
            content=f"Supervision round {state.supervision_round}: {new_sub_queries} follow-up queries",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "iteration": state.iteration,
                "supervision_round": state.supervision_round,
                "follow_ups_added": new_sub_queries,
                "should_continue_gathering": should_continue,
                "overall_coverage": parsed.get("overall_coverage", "unknown"),
            },
        )

    def _build_per_query_coverage(
        self,
        state: DeepResearchState,
    ) -> list[dict[str, Any]]:
        """Build per-sub-query coverage data for supervision assessment.

        For each completed or failed sub-query, computes:
        - Source count (from source_ids on the sub-query)
        - Quality distribution (HIGH/MEDIUM/LOW/UNKNOWN counts)
        - Unique domain count (from source URLs)
        - Findings summary from topic research results

        Args:
            state: Current research state

        Returns:
            List of coverage dicts, one per non-pending sub-query
        """
        coverage: list[dict[str, Any]] = []

        # Build lookup for topic research results by sub_query_id
        topic_results_by_sq: dict[str, Any] = {}
        for tr in state.topic_research_results:
            topic_results_by_sq[tr.sub_query_id] = tr

        for sq in state.sub_queries:
            if sq.status == "pending":
                continue

            # Count sources linked to this sub-query
            sq_sources = [s for s in state.sources if s.sub_query_id == sq.id]
            source_count = len(sq_sources)

            # Quality distribution
            quality_dist: dict[str, int] = {
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0,
                "UNKNOWN": 0,
            }
            for s in sq_sources:
                quality_key = s.quality.value.upper() if s.quality else "UNKNOWN"
                if quality_key in quality_dist:
                    quality_dist[quality_key] += 1
                else:
                    quality_dist["UNKNOWN"] += 1

            # Unique domains
            domains: set[str] = set()
            for s in sq_sources:
                if s.url:
                    try:
                        parsed_url = urlparse(s.url)
                        if parsed_url.netloc:
                            domains.add(parsed_url.netloc)
                    except Exception:
                        pass

            # Findings summary from topic research
            topic_result = topic_results_by_sq.get(sq.id)
            findings_summary = None
            if topic_result and topic_result.per_topic_summary:
                findings_summary = topic_result.per_topic_summary[:500]

            coverage.append(
                {
                    "sub_query_id": sq.id,
                    "query": sq.query,
                    "status": sq.status,
                    "source_count": source_count,
                    "quality_distribution": quality_dist,
                    "unique_domains": len(domains),
                    "domain_list": sorted(domains),
                    "findings_summary": findings_summary,
                }
            )

        return coverage

    def _build_think_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
    ) -> str:
        """Build a gap-analysis-only prompt for the think step.

        This is the think-tool equivalent: it forces the LLM to explicitly
        reason through findings before acting. The prompt asks for structured
        gap analysis WITHOUT producing follow-up queries — that happens in the
        separate act step (coverage assessment).

        The think step articulates:
        - What was found per sub-query
        - What domains/perspectives are represented
        - What perspectives or information gaps exist
        - What specific types of information would fill those gaps

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage from _build_per_query_coverage

        Returns:
            Think prompt string for gap analysis
        """
        parts = [
            f"# Research Gap Analysis\n",
            f"**Original Query:** {state.original_query}\n",
        ]

        if state.research_brief:
            brief_excerpt = state.research_brief[:500]
            parts.append(f"**Research Brief:** {brief_excerpt}\n")

        parts.append(f"**Iteration:** {state.iteration}/{state.max_iterations}")
        parts.append(f"**Supervision Round:** {state.supervision_round + 1}/{state.max_supervision_rounds}")
        parts.append(f"**Total Sources:** {len(state.sources)}\n")

        # Per-sub-query findings and coverage
        if coverage_data:
            parts.append("## Per-Sub-Query Coverage\n")
            for entry in coverage_data:
                parts.append(f"### {entry['query']}")
                parts.append(f"- **Status:** {entry['status']}")
                parts.append(f"- **Sources found:** {entry['source_count']}")
                qd = entry["quality_distribution"]
                parts.append(
                    f"- **Quality:** HIGH={qd['HIGH']}, MEDIUM={qd['MEDIUM']}, "
                    f"LOW={qd['LOW']}"
                )
                parts.append(f"- **Domains:** {', '.join(entry['domain_list']) if entry['domain_list'] else 'none'}")
                if entry.get("findings_summary"):
                    parts.append(f"- **Findings:** {entry['findings_summary']}")
                parts.append("")

        parts.extend([
            "## Instructions\n",
            "Analyze the research coverage above. For EACH sub-query, articulate:",
            "1. What key information was found",
            "2. What domains and perspectives are represented",
            "3. What specific information gaps exist",
            "4. What types of sources or angles would fill those gaps\n",
            "Then provide an overall assessment of:",
            "- Which research dimensions are well-covered",
            "- Which dimensions are missing or underrepresented",
            "- What specific knowledge gaps, if addressed, would most improve the research\n",
            "DO NOT generate follow-up queries. Focus ONLY on analysis of what exists and what's missing.",
            "Be specific: name exact topics, perspectives, or data types that are absent.\n",
            "Respond in plain text with clear section headings.",
        ])

        return "\n".join(parts)

    def _build_think_system_prompt(self) -> str:
        """Build system prompt for the think step LLM call.

        Returns:
            System prompt instructing analytical gap assessment
        """
        return (
            "You are a research gap analyst. Your task is to evaluate the "
            "coverage quality of completed research and identify specific "
            "information gaps. You do NOT generate follow-up queries — you "
            "only analyze what has been found and what is missing.\n\n"
            "Be specific and concise. Name exact topics, perspectives, data "
            "types, or source categories that are absent. Your analysis will "
            "be used by a separate process to generate targeted follow-up queries."
        )

    def _build_supervision_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for coverage assessment.

        Instructs the LLM to evaluate research coverage and return
        structured JSON with follow-up query recommendations.

        Args:
            state: Current research state (reserved for state-aware prompts)

        Returns:
            System prompt string
        """
        _ = state
        return """You are a research supervisor. Your task is to assess the coverage quality of completed research sub-queries and recommend follow-up queries to fill gaps.

Your response MUST be valid JSON with this exact structure:
{
    "overall_coverage": "sufficient|partial|insufficient",
    "per_query_assessment": [
        {
            "sub_query_id": "subq-xxx",
            "coverage": "sufficient|partial|insufficient",
            "rationale": "Why this query's coverage is at this level"
        }
    ],
    "follow_up_queries": [
        {
            "query": "A specific, focused follow-up query",
            "rationale": "What gap this query fills",
            "priority": 2
        }
    ],
    "should_continue_gathering": true,
    "rationale": "Overall assessment of coverage status"
}

Guidelines:
- "sufficient" coverage = 2+ quality sources from diverse domains with relevant findings
- "partial" coverage = some sources but missing key aspects or lacking diversity
- "insufficient" coverage = too few sources, low quality, or missing critical information
- Follow-up queries must be MORE SPECIFIC than original sub-queries (drill down, not repeat)
- Maximum 3 follow-up queries per round
- Do NOT generate queries that duplicate existing sub-queries (check the list provided)
- Set should_continue_gathering=true ONLY if follow-up queries are provided AND coverage is not sufficient
- If overall coverage is "sufficient", set should_continue_gathering=false even if minor gaps exist

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_supervision_user_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        think_output: Optional[str] = None,
    ) -> str:
        """Build user prompt with research context and coverage data.

        Includes the original query, research brief, per-query coverage
        table, existing sub-queries for dedup, round progress, and
        optionally the think-step gap analysis to ground follow-up generation.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage from _build_per_query_coverage
            think_output: Optional gap analysis from the think step. When
                provided, included as a ``<gap_analysis>`` section so the LLM
                generates follow-up queries grounded in explicit reasoning.

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"# Research Query\n{state.original_query}",
            "",
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Supervision round: {state.supervision_round + 1}/{state.max_supervision_rounds}",
            f"- Completed sub-queries: {len(state.completed_sub_queries())}",
            f"- Failed sub-queries: {len(state.failed_sub_queries())}",
            f"- Pending sub-queries: {len(state.pending_sub_queries())}",
            f"- Total sources: {len(state.sources)}",
            "",
        ]

        # Per-query coverage table
        if coverage_data:
            prompt_parts.append("## Per-Query Coverage")
            for entry in coverage_data:
                prompt_parts.append(f"\n### Sub-Query: {entry['sub_query_id']}")
                prompt_parts.append(f"**Query:** {entry['query']}")
                prompt_parts.append(f"**Status:** {entry['status']}")
                prompt_parts.append(f"**Sources:** {entry['source_count']}")
                qd = entry["quality_distribution"]
                prompt_parts.append(
                    f"**Quality:** HIGH={qd['HIGH']}, MEDIUM={qd['MEDIUM']}, "
                    f"LOW={qd['LOW']}, UNKNOWN={qd['UNKNOWN']}"
                )
                prompt_parts.append(f"**Unique domains:** {entry['unique_domains']}")
                if entry.get("findings_summary"):
                    prompt_parts.append(f"**Findings:** {entry['findings_summary']}")
            prompt_parts.append("")

        # Existing sub-queries for dedup reference
        prompt_parts.append("## Existing Sub-Queries (DO NOT duplicate these)")
        for sq in state.sub_queries:
            prompt_parts.append(f"- [{sq.status}] {sq.query}")
        prompt_parts.append("")

        # Think-step gap analysis (grounds follow-up query generation)
        if think_output:
            prompt_parts.extend([
                "## Gap Analysis (from prior deliberation step)",
                "",
                "<gap_analysis>",
                think_output.strip(),
                "</gap_analysis>",
                "",
                "Use the gap analysis above to generate TARGETED follow-up queries "
                "that address the specific gaps identified. Each follow-up query "
                "should directly correspond to a gap mentioned in the analysis.",
                "",
            ])

        # Instructions
        prompt_parts.extend(
            [
                "## Instructions",
                "1. Assess the coverage quality for each completed sub-query",
                "2. Determine overall coverage status",
                "3. If coverage is insufficient or partial, generate specific follow-up queries",
                "4. Follow-up queries should drill deeper, not repeat existing queries",
            ]
        )
        if think_output:
            prompt_parts.append(
                "5. Each follow-up query MUST reference a specific gap from the gap analysis above"
            )
        prompt_parts.extend(["", "Return your assessment as JSON."])

        return "\n".join(prompt_parts)

    def _parse_supervision_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured supervision data.

        Extracts JSON from the response, validates the schema, and
        deduplicates follow-up queries against existing sub-queries
        using case-insensitive exact match.

        Args:
            content: Raw LLM response content
            state: Current research state (for dedup against existing sub-queries)

        Returns:
            Dict with coverage assessment and follow-up queries
        """
        result: dict[str, Any] = {
            "overall_coverage": "unknown",
            "per_query_assessment": [],
            "follow_up_queries": [],
            "should_continue_gathering": False,
            "rationale": "",
        }

        if not content:
            return result

        # Extract JSON from the response
        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in supervision response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from supervision response: %s", e)
            return result

        # Parse overall coverage
        overall = data.get("overall_coverage", "unknown")
        if overall in ("sufficient", "partial", "insufficient"):
            result["overall_coverage"] = overall
        else:
            result["overall_coverage"] = "unknown"

        # Parse per-query assessment
        raw_assessments = data.get("per_query_assessment", [])
        if isinstance(raw_assessments, list):
            for a in raw_assessments:
                if not isinstance(a, dict):
                    continue
                result["per_query_assessment"].append(
                    {
                        "sub_query_id": a.get("sub_query_id", ""),
                        "coverage": a.get("coverage", "unknown"),
                        "rationale": a.get("rationale", ""),
                    }
                )

        # Parse follow-up queries with dedup
        existing_queries_lower = {sq.query.lower().strip() for sq in state.sub_queries}
        raw_follow_ups = data.get("follow_up_queries", [])
        if isinstance(raw_follow_ups, list):
            for fq in raw_follow_ups:
                if not isinstance(fq, dict):
                    continue
                query = fq.get("query", "").strip()
                if not query:
                    continue
                # Case-insensitive dedup against existing sub-queries
                if query.lower().strip() in existing_queries_lower:
                    logger.debug("Supervision: deduped follow-up query: %s", query)
                    continue
                result["follow_up_queries"].append(
                    {
                        "query": query,
                        "rationale": fq.get("rationale", ""),
                        "priority": min(max(int(fq.get("priority", 2)), 1), 10),
                    }
                )
                # Also add to dedup set to prevent duplicates within the same batch
                existing_queries_lower.add(query.lower().strip())

        # Cap follow-ups
        result["follow_up_queries"] = result["follow_up_queries"][:_MAX_FOLLOW_UPS_PER_ROUND]

        # Parse should_continue_gathering
        result["should_continue_gathering"] = bool(data.get("should_continue_gathering", False))

        # If no follow-ups were generated, don't continue gathering
        if not result["follow_up_queries"]:
            result["should_continue_gathering"] = False

        # Parse rationale
        result["rationale"] = data.get("rationale", "")

        return result

    def _assess_coverage_heuristic(
        self,
        state: DeepResearchState,
        min_sources: int,
    ) -> dict[str, Any]:
        """Assess coverage using simple heuristics (no LLM call).

        Used as a fallback when the LLM call fails. Checks whether each
        completed sub-query has at least ``min_sources`` linked sources.

        Returns ``should_continue_gathering=False`` conservatively — the
        heuristic won't generate follow-up queries, so looping back to
        gathering would be pointless without new queries.

        Args:
            state: Current research state
            min_sources: Minimum sources per sub-query for "sufficient" coverage

        Returns:
            Dict with coverage assessment and should_continue_gathering flag
        """
        completed = state.completed_sub_queries()
        if not completed:
            return {
                "overall_coverage": "insufficient",
                "should_continue_gathering": False,
                "queries_assessed": 0,
                "queries_sufficient": 0,
            }

        sufficient_count = 0
        for sq in completed:
            sq_sources = [s for s in state.sources if s.sub_query_id == sq.id]
            if len(sq_sources) >= min_sources:
                sufficient_count += 1

        total = len(completed)
        if sufficient_count == total:
            overall = "sufficient"
        elif sufficient_count > 0:
            overall = "partial"
        else:
            overall = "insufficient"

        return {
            "overall_coverage": overall,
            "should_continue_gathering": False,
            "queries_assessed": total,
            "queries_sufficient": sufficient_count,
        }
