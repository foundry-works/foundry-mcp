"""Supervision phase mixin for DeepResearchWorkflow.

Assesses coverage of completed sub-queries and generates follow-up research
directives to fill gaps before proceeding to analysis.

Supports two supervision models (configurable via ``deep_research_delegation_model``):

1. **Delegation model** (default, Phase 4 PLAN): The supervisor generates
   paragraph-length ``ResearchDirective`` objects targeting specific gaps.
   Directives are executed as parallel topic researchers within the supervision
   phase itself — no re-entry into the GATHERING phase is needed. This mirrors
   open_deep_research's ``ConductResearch`` supervisor pattern.

2. **Query-generation model** (fallback): The supervisor generates single-sentence
   follow-up queries appended to the sub-query list. The workflow then loops back
   to GATHERING to execute them. This is the original model preserved for backward
   compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ResearchDirective,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.sources import SubQuery
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    extract_json,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)

# Maximum follow-up queries the supervisor can generate per round (query-generation model)
_MAX_FOLLOW_UPS_PER_ROUND = 3

# Maximum directives the supervisor can generate per round (delegation model)
# Actual cap also bounded by config.deep_research_max_concurrent_research_units
_MAX_DIRECTIVES_PER_ROUND = 5


class SupervisionPhaseMixin:
    """Supervision phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    - _execute_topic_research_async() (from TopicResearchMixin, for delegation)
    - _get_search_provider() (from GatheringPhaseMixin, for delegation)
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...
        async def _execute_topic_research_async(self, *args: Any, **kwargs: Any) -> Any: ...
        def _get_search_provider(self, provider_name: str) -> Any: ...
        def _get_tavily_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...

    # ==================================================================
    # Main entry point — dispatches to delegation or query-generation
    # ==================================================================

    async def _execute_supervision_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute supervision phase: assess coverage and fill gaps.

        Dispatches to either the delegation model (default) or the legacy
        query-generation model based on ``deep_research_delegation_model`` config.

        Args:
            state: Current research state with completed sub-queries
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with metadata["should_continue_gathering"] flag
        """
        use_delegation = getattr(self.config, "deep_research_delegation_model", True)

        if use_delegation:
            return await self._execute_supervision_delegation_async(
                state, provider_id, timeout,
            )
        else:
            return await self._execute_supervision_query_generation_async(
                state, provider_id, timeout,
            )

    # ==================================================================
    # Delegation model (Phase 4 PLAN)
    # ==================================================================

    async def _execute_supervision_delegation_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute supervision via directive-based delegation.

        Implements the multi-step supervision loop:
        1. **Think**: Analyze compressed findings, identify gaps
        2. **Delegate**: Generate ResearchDirective objects from gap analysis
        3. **Execute**: Spawn parallel topic researchers for directives
        4. **Compress**: Inline compression of new results (via existing infra)
        5. **Assess**: Evaluate coverage and decide continue/complete

        The loop runs within a single supervision phase call. When delegation
        produces new research, the results are merged into state directly —
        no re-entry into the GATHERING phase is needed.

        Args:
            state: Current research state
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with should_continue_gathering=False (delegation
            handles its own gathering internally)
        """
        min_sources = getattr(
            self.config,
            "deep_research_supervision_min_sources_per_query",
            2,
        )

        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "supervision",
                "model": "delegation",
                "iteration": state.iteration,
                "supervision_round": state.supervision_round,
                "task_id": state.id,
            },
        )

        # --- Delegation loop ---
        # Each iteration: think → delegate → execute → assess
        # Bounded by max_supervision_rounds.
        total_directives_executed = 0
        total_new_sources = 0

        while state.supervision_round < state.max_supervision_rounds:
            self._check_cancellation(state)

            logger.info(
                "Supervision delegation round %d/%d: %d completed sub-queries, %d sources",
                state.supervision_round + 1,
                state.max_supervision_rounds,
                len(state.completed_sub_queries()),
                len(state.sources),
            )

            # Build coverage data
            coverage_data = self._build_per_query_coverage(state)

            # --- Heuristic early-exit (round > 0) ---
            if state.supervision_round > 0:
                heuristic = self._assess_coverage_heuristic(state, min_sources)
                if not heuristic["should_continue_gathering"]:
                    logger.info(
                        "Supervision delegation: heuristic sufficient at round %d, advancing",
                        state.supervision_round,
                    )
                    self._write_audit_event(
                        state,
                        "supervision_result",
                        data={
                            "reason": "heuristic_sufficient",
                            "model": "delegation",
                            "supervision_round": state.supervision_round,
                            "coverage_summary": heuristic,
                        },
                    )
                    history = state.metadata.setdefault("supervision_history", [])
                    history.append({
                        "round": state.supervision_round,
                        "method": "delegation_heuristic",
                        "should_continue_gathering": False,
                        "directives_executed": 0,
                        "overall_coverage": "sufficient",
                    })
                    state.supervision_round += 1
                    break

            # --- Step 1: Think (gap analysis) ---
            think_output = await self._supervision_think_step(
                state, coverage_data, timeout,
            )

            # --- Step 2: Delegate (generate directives) ---
            self._check_cancellation(state)
            directives, research_complete = await self._supervision_delegate_step(
                state, coverage_data, think_output, provider_id, timeout,
            )

            # Check for ResearchComplete signal
            if research_complete:
                logger.info(
                    "Supervision delegation: ResearchComplete signal at round %d",
                    state.supervision_round,
                )
                history = state.metadata.setdefault("supervision_history", [])
                history.append({
                    "round": state.supervision_round,
                    "method": "delegation_complete",
                    "should_continue_gathering": False,
                    "directives_generated": 0,
                    "overall_coverage": "sufficient",
                    "think_output": think_output,
                })
                state.supervision_round += 1
                break

            if not directives:
                logger.info(
                    "Supervision delegation: no directives generated at round %d, advancing",
                    state.supervision_round,
                )
                history = state.metadata.setdefault("supervision_history", [])
                history.append({
                    "round": state.supervision_round,
                    "method": "delegation_no_directives",
                    "should_continue_gathering": False,
                    "directives_generated": 0,
                    "overall_coverage": "partial",
                    "think_output": think_output,
                })
                state.supervision_round += 1
                break

            # Store directives for audit
            state.directives.extend(directives)

            # --- Step 3: Execute directives as parallel topic researchers ---
            self._check_cancellation(state)
            directive_results = await self._execute_directives_async(
                state, directives, timeout,
            )

            round_new_sources = sum(r.sources_found for r in directive_results)
            total_new_sources += round_new_sources
            total_directives_executed += len(directive_results)

            # --- Step 4: Think-after-results (assess what was learned) ---
            post_think_output: Optional[str] = None
            if directive_results:
                post_coverage_data = self._build_per_query_coverage(state)
                post_think_output = await self._supervision_think_step(
                    state, post_coverage_data, timeout,
                )

            # --- Step 5: Record history and advance round ---
            history = state.metadata.setdefault("supervision_history", [])
            history.append({
                "round": state.supervision_round,
                "method": "delegation",
                "should_continue_gathering": True,
                "directives_generated": len(directives),
                "directives_executed": len(directive_results),
                "new_sources": round_new_sources,
                "think_output": think_output,
                "post_execution_think": post_think_output,
                "directive_topics": [d.research_topic[:100] for d in directives],
            })

            state.supervision_round += 1
            self.memory.save_deep_research(state)

            # If no new sources were found this round, stop delegating
            if round_new_sources == 0:
                logger.info(
                    "Supervision delegation: no new sources in round %d, stopping",
                    state.supervision_round,
                )
                break

        # Save final state
        self.memory.save_deep_research(state)

        self._write_audit_event(
            state,
            "supervision_result",
            data={
                "model": "delegation",
                "supervision_round": state.supervision_round,
                "total_directives_executed": total_directives_executed,
                "total_new_sources": total_new_sources,
                "should_continue_gathering": False,
            },
        )

        logger.info(
            "Supervision delegation complete: %d rounds, %d directives, %d new sources",
            state.supervision_round,
            total_directives_executed,
            total_new_sources,
        )

        finalize_phase(self, state, "supervision", phase_start_time)

        return WorkflowResult(
            success=True,
            content=(
                f"Supervision delegation: {state.supervision_round} rounds, "
                f"{total_directives_executed} directives, {total_new_sources} new sources"
            ),
            metadata={
                "research_id": state.id,
                "iteration": state.iteration,
                "supervision_round": state.supervision_round,
                "should_continue_gathering": False,
                "total_directives_executed": total_directives_executed,
                "total_new_sources": total_new_sources,
                "model": "delegation",
            },
        )

    # ------------------------------------------------------------------
    # First-round decomposition detection
    # ------------------------------------------------------------------

    def _is_first_round_decomposition(self, state: DeepResearchState) -> bool:
        """Check if this is a first-round supervisor-owned decomposition.

        Returns True when the supervisor should perform initial query
        decomposition (replacing the PLANNING phase) rather than gap-driven
        delegation.

        Conditions:
        - ``deep_research_supervisor_owned_decomposition`` config is True
        - ``state.supervision_round == 0``
        - No prior topic research results exist
        """
        if not getattr(self.config, "deep_research_supervisor_owned_decomposition", True):
            return False
        if state.supervision_round != 0:
            return False
        if state.topic_research_results:
            return False
        return True

    # ------------------------------------------------------------------
    # Delegation sub-steps
    # ------------------------------------------------------------------

    async def _supervision_think_step(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        timeout: float,
    ) -> Optional[str]:
        """Run the think step: gap analysis or first-round decomposition strategy.

        For first-round supervisor-owned decomposition (round 0, no prior
        research), produces a decomposition strategy. For subsequent rounds,
        produces gap analysis.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage data
            timeout: Request timeout

        Returns:
            Think output text, or None if the step failed
        """
        is_first_round = self._is_first_round_decomposition(state)
        if is_first_round:
            think_prompt = self._build_first_round_think_prompt(state)
            think_system = self._build_first_round_think_system_prompt()
        else:
            think_prompt = self._build_think_prompt(state, coverage_data)
            think_system = self._build_think_system_prompt()

        self._check_cancellation(state)

        think_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_think",
            system_prompt=think_system,
            user_prompt=think_prompt,
            provider_id=None,
            model=None,
            temperature=0.2,
            timeout=getattr(self.config, "deep_research_reflection_timeout", 60.0),
            role="reflection",
        )
        if isinstance(think_result, WorkflowResult):
            logger.warning(
                "Supervision think step failed, proceeding without gap analysis: %s",
                think_result.error,
            )
            return None

        think_output = think_result.result.content
        logger.info(
            "Supervision think step completed: %d chars, %s tokens",
            len(think_output or ""),
            think_result.result.tokens_used,
        )
        return think_output

    async def _supervision_delegate_step(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        think_output: Optional[str],
        provider_id: Optional[str],
        timeout: float,
    ) -> tuple[list[ResearchDirective], bool]:
        """Generate research directives from gap or decomposition analysis.

        For first-round supervisor-owned decomposition, generates initial
        research directives that decompose the query (replacing PLANNING).
        For subsequent rounds, generates gap-driven follow-up directives.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage data
            think_output: Gap analysis or decomposition strategy from think step
            provider_id: LLM provider to use
            timeout: Request timeout

        Returns:
            Tuple of (directives list, research_complete flag)
        """
        is_first_round = self._is_first_round_decomposition(state)
        if is_first_round:
            system_prompt = self._build_first_round_delegation_system_prompt()
            user_prompt = self._build_first_round_delegation_user_prompt(
                state, think_output,
            )
        else:
            system_prompt = self._build_delegation_system_prompt()
            user_prompt = self._build_delegation_user_prompt(
                state, coverage_data, think_output,
            )

        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.supervision_provider,
            model=state.supervision_model,
            temperature=0.3,
            timeout=timeout,
            role="delegation",
        )

        if isinstance(call_result, WorkflowResult):
            logger.warning(
                "Supervision delegation LLM call failed: %s. No directives generated.",
                call_result.error,
            )
            return [], False

        directives, research_complete = self._parse_delegation_response(
            call_result.result.content, state,
        )

        self._write_audit_event(
            state,
            "supervision_delegation",
            data={
                "provider_id": call_result.result.provider_id,
                "model_used": call_result.result.model_used,
                "tokens_used": call_result.result.tokens_used,
                "directives_generated": len(directives),
                "research_complete": research_complete,
                "directive_topics": [d.research_topic[:100] for d in directives],
                "is_first_round_decomposition": is_first_round,
            },
        )

        return directives, research_complete

    async def _execute_directives_async(
        self,
        state: DeepResearchState,
        directives: list[ResearchDirective],
        timeout: float,
    ) -> list[TopicResearchResult]:
        """Execute research directives as parallel topic researchers.

        Converts each directive into a SubQuery and spawns parallel
        topic researchers using the existing gathering infrastructure.
        Results (sources, topic research results) are merged into state.

        Args:
            state: Current research state
            directives: Research directives to execute
            timeout: Per-operation timeout

        Returns:
            List of TopicResearchResult from directive execution
        """
        max_concurrent = getattr(
            self.config, "deep_research_max_concurrent_research_units", 5,
        )
        topic_max_searches = getattr(
            self.config, "deep_research_topic_max_tool_calls", 10,
        )
        max_sources_per_provider = max(2, state.max_sources_per_query // max(1, len(directives)))

        # Sort directives by priority (1=critical first)
        sorted_directives = sorted(directives, key=lambda d: d.priority)

        # Initialize search providers
        provider_names = getattr(
            self.config,
            "deep_research_providers",
            ["tavily", "google", "semantic_scholar"],
        )
        available_providers = []
        for name in provider_names:
            provider = self._get_search_provider(name)
            if provider is not None:
                available_providers.append(provider)

        if not available_providers:
            logger.warning("No search providers available for directive execution")
            return []

        # Shared dedup state
        seen_urls: set[str] = {s.url for s in state.sources if s.url}
        seen_titles: dict[str, str] = {}
        from foundry_mcp.core.research.workflows.deep_research.source_quality import _normalize_title
        for source in state.sources:
            normalized = _normalize_title(source.title)
            if normalized and len(normalized) > 20:
                seen_titles.setdefault(normalized, source.url or "")

        semaphore = asyncio.Semaphore(max_concurrent)
        state_lock = asyncio.Lock()

        # Create sub-queries from directives
        directive_sub_queries: list[SubQuery] = []
        for directive in sorted_directives:
            sq = state.add_sub_query(
                query=directive.research_topic,
                rationale=f"Delegation directive (round {directive.supervision_round}): {directive.perspective}",
                priority=directive.priority,
            )
            directive_sub_queries.append(sq)

        logger.info(
            "Executing %d directives as parallel topic researchers (max_concurrent=%d)",
            len(directive_sub_queries),
            max_concurrent,
        )

        self._write_audit_event(
            state,
            "directive_execution_start",
            data={
                "directive_count": len(directive_sub_queries),
                "max_concurrent": max_concurrent,
                "available_providers": [p.get_provider_name() for p in available_providers],
            },
        )

        # Spawn parallel topic researchers
        async def run_directive_researcher(sq: SubQuery) -> TopicResearchResult:
            try:
                return await self._execute_topic_research_async(
                    sub_query=sq,
                    state=state,
                    available_providers=available_providers,
                    max_searches=topic_max_searches,
                    max_sources_per_provider=max_sources_per_provider,
                    timeout=timeout,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                    semaphore=semaphore,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Directive researcher failed for sub-query %s: %s. Non-fatal.",
                    sq.id,
                    exc,
                )
                sq.mark_failed(f"Directive execution failed: {exc}")
                return TopicResearchResult(
                    sub_query_id=sq.id,
                    sources_found=0,
                    completion_rationale=f"Failed: {exc}",
                )

        tasks = [run_directive_researcher(sq) for sq in directive_sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Merge results into state
        for result in results:
            state.topic_research_results.append(result)

        self.memory.save_deep_research(state)

        successful = [r for r in results if r.sources_found > 0]
        total_sources = sum(r.sources_found for r in results)
        logger.info(
            "Directive execution complete: %d/%d successful, %d total new sources",
            len(successful),
            len(results),
            total_sources,
        )

        self._write_audit_event(
            state,
            "directive_execution_complete",
            data={
                "directives_total": len(results),
                "directives_successful": len(successful),
                "total_new_sources": total_sources,
            },
        )

        return list(results)

    # ------------------------------------------------------------------
    # Delegation prompts
    # ------------------------------------------------------------------

    def _build_delegation_system_prompt(self) -> str:
        """Build system prompt for the delegation step.

        Returns:
            System prompt instructing directive generation
        """
        return """You are a research lead delegating tasks to specialized researchers. Your task is to analyze research gaps and generate detailed research directives.

Your response MUST be valid JSON with this exact structure:
{
    "research_complete": false,
    "directives": [
        {
            "research_topic": "Detailed paragraph-length description of what to investigate...",
            "perspective": "What angle or perspective to approach from",
            "evidence_needed": "What specific evidence, data, or sources to seek",
            "priority": 1
        }
    ],
    "rationale": "Why these directives were chosen"
}

Guidelines:
- Set "research_complete" to true ONLY when existing coverage is sufficient across all dimensions
- Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences) specifying:
  - The specific topic to investigate
  - The research approach (compare, investigate, validate, survey, etc.)
  - What the researcher should focus on and what to exclude
- "perspective" should specify the angle: technical, comparative, historical, regulatory, user-focused, etc.
- "evidence_needed" should name concrete evidence types: statistics, case studies, expert opinions, benchmarks, etc.
- "priority": 1=critical gap (blocks report quality), 2=important (improves comprehensiveness), 3=nice-to-have
- Maximum 5 directives per round
- Do NOT duplicate research already covered — target SPECIFIC gaps
- Directives should be complementary, not overlapping — each covers a different dimension

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_delegation_user_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        think_output: Optional[str] = None,
    ) -> str:
        """Build user prompt for directive generation.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage data
            think_output: Gap analysis from think step

        Returns:
            User prompt string
        """
        parts = [
            f"# Research Query\n{state.original_query}",
            "",
        ]

        if state.research_brief and state.research_brief != state.original_query:
            parts.extend([
                "## Research Brief",
                state.research_brief,
                "",
            ])

        parts.extend([
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Supervision round: {state.supervision_round + 1}/{state.max_supervision_rounds}",
            f"- Completed sub-queries: {len(state.completed_sub_queries())}",
            f"- Total sources: {len(state.sources)}",
            "",
        ])

        # Per-query coverage with compressed findings
        if coverage_data:
            parts.append("## Current Research Coverage")
            for entry in coverage_data:
                parts.append(f"\n### {entry['query']}")
                parts.append(f"**Sources:** {entry['source_count']} | **Domains:** {entry['unique_domains']}")
                if entry.get("compressed_findings_excerpt"):
                    parts.append(f"**Key findings:**\n{entry['compressed_findings_excerpt']}")
                elif entry.get("findings_summary"):
                    parts.append(f"**Summary:** {entry['findings_summary']}")
            parts.append("")

        # Gap analysis from think step
        if think_output:
            parts.extend([
                "## Gap Analysis",
                "",
                "<gap_analysis>",
                think_output.strip(),
                "</gap_analysis>",
                "",
                "Generate research directives that DIRECTLY address the gaps identified above.",
                "Each directive should target a specific gap with a detailed research plan.",
                "",
            ])

        # Previously executed directives (to avoid repetition)
        if state.directives:
            parts.append("## Previously Executed Directives (DO NOT repeat)")
            for d in state.directives[-10:]:  # Last 10 for context
                parts.append(f"- [P{d.priority}] {d.research_topic[:120]}")
            parts.append("")

        parts.extend([
            "## Instructions",
            "1. Analyze the current coverage and gap analysis",
            "2. If all research dimensions are well-covered, set research_complete=true",
            "3. Otherwise, generate 1-5 detailed research directives targeting specific gaps",
            "4. Each directive should be a paragraph-length research assignment",
            "5. Prioritize critical gaps (priority 1) over nice-to-have improvements (priority 3)",
            "",
            "Return your response as JSON.",
        ])

        return "\n".join(parts)

    def _parse_delegation_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> tuple[list[ResearchDirective], bool]:
        """Parse LLM response into research directives.

        Args:
            content: Raw LLM response
            state: Current research state

        Returns:
            Tuple of (directives list, research_complete flag)
        """
        if not content:
            return [], False

        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in delegation response")
            # Graceful fallback: try to create a single directive from the think output
            return [], False

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse delegation JSON: %s", e)
            return [], False

        # Check for research complete signal
        research_complete = bool(data.get("research_complete", False))
        if research_complete:
            return [], True

        # Parse directives
        max_units = getattr(
            self.config, "deep_research_max_concurrent_research_units", 5,
        )
        cap = min(max_units, _MAX_DIRECTIVES_PER_ROUND)

        raw_directives = data.get("directives", [])
        if not isinstance(raw_directives, list):
            logger.warning("Delegation response 'directives' is not a list")
            return [], False

        directives: list[ResearchDirective] = []
        for d in raw_directives[:cap]:
            if not isinstance(d, dict):
                continue
            topic = d.get("research_topic", "").strip()
            if not topic:
                continue
            # Validate priority
            try:
                priority = min(max(int(d.get("priority", 2)), 1), 3)
            except (ValueError, TypeError):
                priority = 2

            directives.append(ResearchDirective(
                research_topic=topic,
                perspective=d.get("perspective", ""),
                evidence_needed=d.get("evidence_needed", ""),
                priority=priority,
                supervision_round=state.supervision_round,
            ))

        return directives, False

    # ==================================================================
    # First-round decomposition prompts (supervisor-owned decomposition)
    # ==================================================================

    def _build_first_round_think_system_prompt(self) -> str:
        """Build system prompt for the first-round decomposition think step.

        On the first supervision round (round 0) when supervisor-owned
        decomposition is active, the think step produces a decomposition
        strategy rather than gap analysis.

        Returns:
            System prompt instructing decomposition strategy generation
        """
        return (
            "You are a research strategist. Your task is to analyze a research "
            "brief and determine the best decomposition strategy for parallel "
            "research. You decide how many parallel researchers to launch, what "
            "angles they should cover, and what priorities to assign.\n\n"
            "Be strategic: simple factual queries may need only 1-2 researchers. "
            "Comparative analyses need one researcher per element being compared. "
            "Complex multi-dimensional topics need 3-5 researchers covering "
            "different facets.\n\n"
            "Before finalizing your strategy, verify:\n"
            "- No two researchers would cover substantially the same ground\n"
            "- No critical perspective is missing for the query type\n"
            "- Each researcher has a specific, actionable focus\n\n"
            "Respond in plain text with clear section headings."
        )

    def _build_first_round_think_prompt(self, state: DeepResearchState) -> str:
        """Build think prompt for first-round decomposition strategy.

        Presents the research brief and asks for a decomposition plan
        before generating directives.

        Args:
            state: Current research state with research_brief

        Returns:
            Think prompt string for decomposition strategy
        """
        from datetime import datetime, timezone

        parts = [
            "# Research Decomposition Strategy\n",
            f"**Research Query:** {state.original_query}\n",
        ]

        if state.research_brief and state.research_brief != state.original_query:
            parts.append(f"**Research Brief:**\n{state.research_brief}\n")

        if state.clarification_constraints:
            parts.append("**Clarification Constraints:**")
            for key, value in state.clarification_constraints.items():
                parts.append(f"- {key}: {value}")
            parts.append("")

        if state.system_prompt:
            parts.append(f"**Additional Context:** {state.system_prompt}\n")

        parts.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n")

        parts.extend([
            "## Instructions\n",
            "You are given a research brief. Determine how to decompose this "
            "into parallel research tasks.\n",
            "Analyze the query and decide:",
            "1. **Query type**: Is this a simple factual query, a comparison, "
            "a list/ranking, or a complex multi-dimensional topic?",
            "2. **Decomposition strategy**: How many parallel researchers are "
            "needed and what angle should each cover?",
            "3. **Priorities**: Which research angles are critical (must-have) "
            "vs. important (improves comprehensiveness) vs. nice-to-have?",
            "4. **Self-critique**: Verify no redundant directives and no "
            "missing perspectives for this query type.\n",
            "Guidelines for researcher count:",
            "- Simple factual queries: 1-2 researchers",
            "- Comparisons: one researcher per comparison element",
            "- Lists/rankings: single researcher if straightforward, or one per "
            "category if complex",
            "- Complex multi-dimensional topics: 3-5 researchers covering "
            "different facets\n",
            "Output your decomposition strategy as structured analysis with "
            "clear headings.",
        ])

        return "\n".join(parts)

    def _build_first_round_delegation_system_prompt(self) -> str:
        """Build system prompt for first-round decomposition delegation.

        Combines the standard delegation format with planning-quality
        decomposition guidance. This replaces the PLANNING phase's
        decomposition rules.

        Returns:
            System prompt instructing initial query decomposition via directives
        """
        return """You are a research lead performing initial query decomposition. Your task is to break down a research query into focused, parallel research directives — each assigned to a specialized researcher.

Your response MUST be valid JSON with this exact structure:
{
    "research_complete": false,
    "directives": [
        {
            "research_topic": "Detailed paragraph-length description of what to investigate...",
            "perspective": "What angle or perspective to approach from",
            "evidence_needed": "What specific evidence, data, or sources to seek",
            "priority": 1
        }
    ],
    "rationale": "Why this decomposition strategy was chosen"
}

Decomposition Guidelines:
- Generate 2-5 directives (aim for 3-4 typically for most queries)
- Bias toward FEWER researchers for simple queries (1-2 directives for straightforward factual questions)
- For COMPARISONS: create one directive per comparison element (e.g., "Product A vs Product B" → one directive for each product)
- For LISTS/RANKINGS: single directive if straightforward; one per category if the list spans diverse domains
- For COMPLEX multi-dimensional topics: 3-5 directives covering different facets (technical, economic, regulatory, user impact, etc.)

Quality Guidelines:
- Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences) specifying:
  - The specific topic or facet to investigate
  - The research approach (compare, investigate, validate, survey, etc.)
  - What the researcher should focus on and what to exclude
- Each directive should be SPECIFIC enough to yield targeted search results
- Directives must cover DISTINCT aspects — no two should investigate substantially the same ground
- "perspective" should specify the angle: technical, comparative, historical, regulatory, user-focused, economic, etc.
- "evidence_needed" should name concrete evidence types: statistics, case studies, expert opinions, benchmarks, official documentation, etc.
- "priority": 1=critical (core to answering the query), 2=important (improves comprehensiveness), 3=nice-to-have (supplementary context)

Self-Critique Checklist (verify before responding):
- Are any directives redundant? If so, merge them.
- Is any critical perspective missing for this type of query?
- Are directives specific enough, or are they too broad/vague?

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_first_round_delegation_user_prompt(
        self,
        state: DeepResearchState,
        think_output: Optional[str] = None,
    ) -> str:
        """Build user prompt for first-round decomposition delegation.

        Args:
            state: Current research state with research_brief
            think_output: Decomposition strategy from think step

        Returns:
            User prompt string
        """
        parts = [
            f"# Research Query\n{state.original_query}",
            "",
        ]

        if state.research_brief and state.research_brief != state.original_query:
            parts.extend([
                "## Research Brief",
                state.research_brief,
                "",
            ])

        if state.clarification_constraints:
            parts.append("## Clarification Constraints")
            for key, value in state.clarification_constraints.items():
                parts.append(f"- {key}: {value}")
            parts.append("")

        if state.system_prompt:
            parts.extend([
                "## Additional Context",
                state.system_prompt,
                "",
            ])

        # Decomposition strategy from think step
        if think_output:
            parts.extend([
                "## Decomposition Strategy",
                "",
                "<decomposition_strategy>",
                think_output.strip(),
                "</decomposition_strategy>",
                "",
                "Generate research directives that implement the decomposition "
                "strategy above. Each directive should be a detailed, self-contained "
                "research assignment for a specialized researcher.",
                "",
            ])

        parts.extend([
            "## Instructions",
            "1. Decompose the research query into 2-5 focused research directives",
            "2. Each directive should target a distinct aspect of the query",
            "3. Each directive should be specific enough to yield targeted results",
            "4. Prioritize: 1=critical to the core question, 2=important for comprehensiveness, 3=supplementary",
            "5. Verify no redundant directives and no missing critical perspectives",
            "",
            "Return your response as JSON.",
        ])

        return "\n".join(parts)

    # ==================================================================
    # Query-generation model (legacy fallback)
    # ==================================================================

    async def _execute_supervision_query_generation_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute supervision via follow-up query generation (legacy model).

        This is the original supervision implementation preserved for backward
        compatibility when ``deep_research_delegation_model=False``.

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
                "model": "query_generation",
                "iteration": state.iteration,
                "supervision_round": state.supervision_round,
                "task_id": state.id,
            },
        )

        # --- Think step: deliberate on gaps before generating follow-ups ---
        think_output: Optional[str] = None
        if state.supervision_round == 0:
            think_output = await self._supervision_think_step(
                state, coverage_data, timeout,
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

    # ==================================================================
    # Shared helpers (used by both models)
    # ==================================================================

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
        - Compressed findings excerpt (when inline compression is available)

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
            compressed_findings_excerpt = None
            if topic_result:
                if topic_result.per_topic_summary:
                    findings_summary = topic_result.per_topic_summary[:500]
                # Include compressed findings when available (from inline compression)
                if topic_result.compressed_findings:
                    compressed_findings_excerpt = topic_result.compressed_findings[:2000]

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
                    "compressed_findings_excerpt": compressed_findings_excerpt,
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
        """Build system prompt for coverage assessment (query-generation model).

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
- "sufficient" coverage = 2+ quality sources from diverse domains with relevant findings addressing the research brief's dimensions
- "partial" coverage = some sources but missing key aspects, lacking diversity, or leaving dimensions from the research brief unaddressed
- "insufficient" coverage = too few sources, low quality, or missing critical information needed by the research brief
- Follow-up queries must be MORE SPECIFIC than original sub-queries (drill down, not repeat)
- Maximum 3 follow-up queries per round
- Do NOT generate queries that duplicate existing sub-queries (check the list provided)
- Set should_continue_gathering=true ONLY if follow-up queries are provided AND coverage is not sufficient
- If overall coverage is "sufficient", set should_continue_gathering=false even if minor gaps exist

Content Assessment (when compressed findings are provided):
- Evaluate whether the findings SUBSTANTIVELY address the research brief's dimensions, not just whether sources exist
- Identify specific CONTENT gaps where important perspectives, evidence types, or data points are missing
- Consider both quantitative coverage (source count, domain diversity) and qualitative coverage (finding depth, relevance)
- Distinguish between topics where sources all cover the same angle vs. topics with genuinely diverse findings
- When compressed findings are provided, base your assessment primarily on the content, not the source count

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
        ]

        # Include research brief scope for coverage assessment
        if state.research_brief and state.research_brief != state.original_query:
            prompt_parts.extend([
                "## Research Brief (scope boundaries for coverage assessment)",
                state.research_brief,
                "",
            ])

        prompt_parts.extend([
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Supervision round: {state.supervision_round + 1}/{state.max_supervision_rounds}",
            f"- Completed sub-queries: {len(state.completed_sub_queries())}",
            f"- Failed sub-queries: {len(state.failed_sub_queries())}",
            f"- Pending sub-queries: {len(state.pending_sub_queries())}",
            f"- Total sources: {len(state.sources)}",
            "",
        ])

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
                # Include compressed findings when available (from inline compression)
                # This enables content-aware coverage assessment
                if entry.get("compressed_findings_excerpt"):
                    prompt_parts.append(f"**Key findings:**\n{entry['compressed_findings_excerpt']}")
                elif entry.get("findings_summary"):
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
