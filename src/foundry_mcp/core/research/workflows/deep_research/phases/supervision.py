"""Supervision phase mixin for DeepResearchWorkflow.

Assesses coverage of completed sub-queries and generates follow-up research
directives to fill gaps before proceeding to analysis.

Uses the **delegation model**: the supervisor generates paragraph-length
``ResearchDirective`` objects targeting specific gaps.  Directives are executed
as parallel topic researchers within the supervision phase itself — no re-entry
into the GATHERING phase is needed.  This mirrors open_deep_research's
``ConductResearch`` supervisor pattern.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

from foundry_mcp.core.research.models.deep_research import (
    DelegationResponse,
    DeepResearchState,
    ResearchDirective,
    TopicResearchResult,
    parse_delegation_response,
)
from foundry_mcp.core.research.models.sources import SubQuery
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    extract_json,
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    StructuredLLMCallResult,
    execute_llm_call,
    execute_structured_llm_call,
    finalize_phase,
    truncate_supervision_messages,
)

logger = logging.getLogger(__name__)

# Maximum follow-up queries the supervisor can generate per round (legacy path)
_MAX_FOLLOW_UPS_PER_ROUND = 3

# Maximum directives the supervisor can generate per round
# Actual cap also bounded by config.deep_research_max_concurrent_research_units
_MAX_DIRECTIVES_PER_ROUND = 5

# Cap stored directives to bound state serialization size.
# Only the most recent directives are kept; older ones are pruned.
_MAX_STORED_DIRECTIVES = 30

# Cap raw_notes entries to prevent unbounded state growth.
# With 6 rounds × 5 researchers, notes can grow to MBs unchecked.
# When exceeded, oldest entries are dropped and an audit event is logged.
_MAX_RAW_NOTES = 50
_MAX_RAW_NOTES_CHARS = 500_000


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
        async def _compress_single_topic_async(
            self,
            topic_result: TopicResearchResult,
            state: DeepResearchState,
            timeout: float,
        ) -> tuple[int, int, bool]: ...

    # ==================================================================
    # Main entry point
    # ==================================================================

    async def _execute_supervision_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute supervision phase: assess coverage and fill gaps.

        Uses the delegation model: generates ``ResearchDirective`` objects
        targeting specific gaps and executes them as parallel topic researchers.

        Args:
            state: Current research state with completed sub-queries
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with supervision round metadata
        """
        return await self._execute_supervision_delegation_async(
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
            WorkflowResult with supervision round metadata (delegation
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
        # Bounded by max_supervision_rounds and wall-clock timeout.
        total_directives_executed = 0
        total_new_sources = 0

        wall_clock_start = time.monotonic()
        wall_clock_limit: float = getattr(
            self.config, "deep_research_supervision_wall_clock_timeout", 1800.0,
        )

        while state.supervision_round < state.max_supervision_rounds:
            self._check_cancellation(state)

            # --- Wall-clock timeout guard ---
            elapsed = time.monotonic() - wall_clock_start
            if elapsed >= wall_clock_limit:
                logger.warning(
                    "Supervision phase wall-clock timeout: %.0fs elapsed >= %.0fs limit. "
                    "Exiting after %d rounds.",
                    elapsed,
                    wall_clock_limit,
                    state.supervision_round,
                )
                state.metadata["supervision_wall_clock_exit"] = {
                    "elapsed_seconds": round(elapsed, 1),
                    "limit_seconds": wall_clock_limit,
                    "rounds_completed": state.supervision_round,
                }
                self._write_audit_event(
                    state,
                    "supervision_wall_clock_timeout",
                    data={
                        "elapsed_seconds": round(elapsed, 1),
                        "limit_seconds": wall_clock_limit,
                        "rounds_completed": state.supervision_round,
                    },
                )
                break

            logger.info(
                "Supervision delegation round %d/%d: %d completed sub-queries, %d sources",
                state.supervision_round + 1,
                state.max_supervision_rounds,
                len(state.completed_sub_queries()),
                len(state.sources),
            )

            # Apply token-limit guard on supervision message history
            if state.supervision_messages:
                state.supervision_messages = truncate_supervision_messages(
                    state.supervision_messages,
                    model=state.supervision_model,
                )

            # Build coverage data
            coverage_data = self._build_per_query_coverage(state)

            # --- Coverage delta for think-step focus (Phase 5) ---
            coverage_delta: Optional[str] = None
            if state.supervision_round > 0:
                coverage_delta = self._compute_coverage_delta(
                    state, coverage_data, min_sources=min_sources,
                )

            # Store coverage snapshot for use in subsequent rounds
            self._store_coverage_snapshot(state, coverage_data)

            # --- Heuristic early-exit (round > 0) ---
            if state.supervision_round > 0:
                heuristic = self._assess_coverage_heuristic(state, min_sources)
                if not heuristic["should_continue_gathering"]:
                    logger.info(
                        "Supervision delegation: confidence %.2f >= threshold at round %d, advancing",
                        heuristic.get("confidence", 0.0),
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

            # --- Steps 1+2: Think (gap analysis) + Delegate (directives) ---
            # Phase 6: single-call mode merges think+delegate into one LLM
            # call for lower latency. Two-call mode (default) keeps them
            # separate but injects think into message history before delegate.
            use_single_call = getattr(
                self.config, "deep_research_supervision_single_call", False,
            )
            is_first_round = self._is_first_round_decomposition(state)

            if use_single_call and not is_first_round:
                # --- Single-call path (Phase 6, option 6.3) ---
                (
                    think_output, directives, research_complete, delegation_content,
                ) = await self._supervision_combined_think_delegate_step(
                    state, coverage_data, provider_id, timeout,
                )

                # Inject think and delegation into conversation
                if think_output:
                    state.supervision_messages.append({
                        "role": "assistant",
                        "type": "think",
                        "round": state.supervision_round,
                        "content": think_output,
                    })
                if delegation_content:
                    state.supervision_messages.append({
                        "role": "assistant",
                        "type": "delegation",
                        "round": state.supervision_round,
                        "content": delegation_content,
                    })
            else:
                # --- Two-call path (Phase 6, option 6.4 — default) ---
                think_output = await self._supervision_think_step(
                    state, coverage_data, timeout,
                    coverage_delta=coverage_delta,
                )

                # Inject think into conversation BEFORE delegation so the
                # delegation LLM sees the supervisor's gap reasoning as part
                # of the accumulated conversation.
                if think_output:
                    state.supervision_messages.append({
                        "role": "assistant",
                        "type": "think",
                        "round": state.supervision_round,
                        "content": think_output,
                    })

                self._check_cancellation(state)
                directives, research_complete, delegation_content = (
                    await self._supervision_delegate_step(
                        state, coverage_data, think_output, provider_id, timeout,
                    )
                )

                # Capture delegation response as assistant message
                if delegation_content:
                    state.supervision_messages.append({
                        "role": "assistant",
                        "type": "delegation",
                        "round": state.supervision_round,
                        "content": delegation_content,
                    })

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

            # Store directives for audit (capped to limit state serialization growth)
            state.directives.extend(directives)
            state.directives = state.directives[-_MAX_STORED_DIRECTIVES:]

            # --- Step 3: Execute directives as parallel topic researchers ---
            self._check_cancellation(state)
            directive_results = await self._execute_directives_async(
                state, directives, timeout,
            )

            round_new_sources = sum(r.sources_found for r in directive_results)
            total_new_sources += round_new_sources
            total_directives_executed += len(directive_results)

            # --- Inline compression of directive results ---
            # Compress results before appending to supervision messages
            # to reduce message history growth rate (~45% reduction).
            inline_stats = await self._compress_directive_results_inline(
                state, directive_results, timeout,
            )

            # --- Accumulate findings as tool-result messages ---
            for result in directive_results:
                content = result.compressed_findings
                if not content and result.source_ids:
                    content = self._build_directive_fallback_summary(
                        result, state,
                    )
                if content:
                    state.supervision_messages.append({
                        "role": "tool_result",
                        "type": "research_findings",
                        "round": state.supervision_round,
                        "directive_id": result.sub_query_id,
                        "content": content,
                    })

            # --- Aggregate raw notes (Phase 1 ODR alignment) ---
            for result in directive_results:
                if result.raw_notes:
                    state.raw_notes.append(result.raw_notes)

            # --- Trim raw_notes if they exceed the cap ---
            notes_trimmed = 0
            while len(state.raw_notes) > _MAX_RAW_NOTES:
                state.raw_notes.pop(0)
                notes_trimmed += 1
            # Also enforce character budget: drop oldest until under limit
            total_chars = sum(len(n) for n in state.raw_notes)
            while state.raw_notes and total_chars > _MAX_RAW_NOTES_CHARS:
                removed = state.raw_notes.pop(0)
                total_chars -= len(removed)
                notes_trimmed += 1
            if notes_trimmed > 0:
                logger.warning(
                    "Trimmed %d oldest raw_notes entries (count cap=%d, char cap=%d). "
                    "%d entries remain (%d chars).",
                    notes_trimmed,
                    _MAX_RAW_NOTES,
                    _MAX_RAW_NOTES_CHARS,
                    len(state.raw_notes),
                    total_chars,
                )
                self._write_audit_event(
                    state,
                    "raw_notes_trimmed",
                    data={
                        "entries_trimmed": notes_trimmed,
                        "entries_remaining": len(state.raw_notes),
                        "chars_remaining": total_chars,
                        "round": state.supervision_round,
                    },
                )

            # --- Append evidence inventories (Phase 2 ODR alignment) ---
            # Gives the supervisor a compact evidence reference alongside
            # compressed findings, preventing re-investigation of covered topics.
            for result in directive_results:
                if result.raw_notes or result.source_ids:
                    inventory = self._build_evidence_inventory(result, state)
                    if inventory:
                        state.supervision_messages.append({
                            "role": "tool_result",
                            "type": "evidence_inventory",
                            "round": state.supervision_round,
                            "directive_id": result.sub_query_id,
                            "content": inventory,
                        })

            # --- Step 4: Think-after-results (assess what was learned) ---
            post_think_output: Optional[str] = None
            if directive_results:
                post_coverage_data = self._build_per_query_coverage(state)
                # Compute delta against the pre-execution snapshot for this round
                post_delta = self._compute_coverage_delta(
                    state, post_coverage_data, min_sources=min_sources,
                )
                # Update the snapshot with post-execution coverage
                self._store_coverage_snapshot(state, post_coverage_data)
                post_think_output = await self._supervision_think_step(
                    state, post_coverage_data, timeout,
                    coverage_delta=post_delta,
                )
                # Phase 6: Post-execution assessment flows into conversation
                # history so the next round's delegation sees what was learned.
                if post_think_output:
                    state.supervision_messages.append({
                        "role": "assistant",
                        "type": "think",
                        "round": state.supervision_round,
                        "content": post_think_output,
                    })

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
                "inline_compression": inline_stats,
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
        - ``state.supervision_round == 0``
        - No prior topic research results exist
        """
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
        coverage_delta: Optional[str] = None,
    ) -> Optional[str]:
        """Run the think step: gap analysis or first-round decomposition strategy.

        For first-round supervisor-owned decomposition (round 0, no prior
        research), produces a decomposition strategy. For subsequent rounds,
        produces gap analysis.

        When a ``coverage_delta`` is provided (from ``_compute_coverage_delta``),
        it is injected into the think prompt to help the LLM focus on what
        changed since the previous round.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage data
            timeout: Request timeout
            coverage_delta: Optional delta summary for rounds > 0

        Returns:
            Think output text, or None if the step failed
        """
        is_first_round = self._is_first_round_decomposition(state)
        if is_first_round:
            think_prompt = self._build_first_round_think_prompt(state)
            think_system = self._build_first_round_think_system_prompt()
        else:
            think_prompt = self._build_think_prompt(
                state, coverage_data, coverage_delta=coverage_delta,
            )
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
    ) -> tuple[list[ResearchDirective], bool, Optional[str]]:
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
            Tuple of (directives list, research_complete flag, raw LLM response content)
        """
        is_first_round = self._is_first_round_decomposition(state)

        # First-round decomposition uses a 3-call pipeline:
        # generate → critique → revise for higher-quality directives.
        if is_first_round:
            return await self._first_round_decompose_critique_revise(
                state, think_output, provider_id, timeout,
            )

        system_prompt = self._build_delegation_system_prompt()
        user_prompt = self._build_delegation_user_prompt(
            state, coverage_data, think_output,
        )

        # Use structured output parsing with automatic retry on parse failure
        call_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.supervision_provider,
            model=state.supervision_model,
            temperature=0.3,
            timeout=timeout,
            parse_fn=parse_delegation_response,
            role="delegation",
        )

        if isinstance(call_result, WorkflowResult):
            logger.warning(
                "Supervision delegation LLM call failed: %s. No directives generated.",
                call_result.error,
            )
            return [], False, None

        # Structured parse succeeded — extract directives from DelegationResponse
        if call_result.parsed is not None:
            delegation: DelegationResponse = call_result.parsed
            research_complete = delegation.research_complete
            directives = self._apply_directive_caps(delegation.directives, state)
        else:
            # Structured parse exhausted — fall back to legacy manual parsing
            logger.warning(
                "Structured delegation parse failed, falling back to legacy parser",
            )
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
                "is_first_round_decomposition": False,
                "structured_parse": call_result.parsed is not None,
                "parse_retries": call_result.parse_retries,
            },
        )

        # Return raw LLM response content for message accumulation
        raw_content = call_result.result.content
        return directives, research_complete, raw_content

    async def _supervision_combined_think_delegate_step(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        provider_id: Optional[str],
        timeout: float,
    ) -> tuple[Optional[str], list[ResearchDirective], bool, Optional[str]]:
        """Combined think+delegate in a single LLM call (Phase 6).

        Merges gap analysis and directive generation into one conversation
        turn, reducing latency while keeping the think output as an
        explicit section in the response. The LLM first reasons about gaps,
        then produces structured JSON directives.

        This is the single-call alternative to the two-step think → delegate
        flow. Enable via ``deep_research_supervision_single_call`` config.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage data
            provider_id: LLM provider to use
            timeout: Request timeout

        Returns:
            Tuple of (think_output, directives, research_complete, raw_content)
        """
        system_prompt = self._build_combined_think_delegate_system_prompt()
        user_prompt = self._build_combined_think_delegate_user_prompt(
            state, coverage_data,
        )

        call_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_combined_think_delegate",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.supervision_provider,
            model=state.supervision_model,
            temperature=0.3,
            timeout=timeout,
            parse_fn=self._parse_combined_response,
            role="delegation",
        )

        if isinstance(call_result, WorkflowResult):
            logger.warning(
                "Combined think+delegate call failed: %s",
                call_result.error,
            )
            return None, [], False, None

        raw_content = call_result.result.content or ""

        if call_result.parsed is not None:
            think_output, delegation = call_result.parsed
            research_complete = delegation.research_complete
            directives = self._apply_directive_caps(delegation.directives, state)
        else:
            # Parse failed — extract what we can
            logger.warning(
                "Combined response parse failed, attempting fallback extraction",
            )
            think_output = self._extract_gap_analysis_section(raw_content)
            directives, research_complete = self._parse_delegation_response(
                raw_content, state,
            )

        self._write_audit_event(
            state,
            "supervision_combined_think_delegate",
            data={
                "provider_id": call_result.result.provider_id,
                "model_used": call_result.result.model_used,
                "tokens_used": call_result.result.tokens_used,
                "directives_generated": len(directives),
                "research_complete": research_complete,
                "has_think_output": think_output is not None,
                "structured_parse": call_result.parsed is not None,
                "parse_retries": call_result.parse_retries,
            },
        )

        return think_output, directives, research_complete, raw_content

    def _build_combined_think_delegate_system_prompt(self) -> str:
        """Build system prompt for the combined think+delegate step."""
        return """You are a research lead. Your task has two parts:

**Part 1 — Gap Analysis:** Analyze the research coverage and identify specific information gaps. Articulate:
- What was found per sub-query
- What domains and perspectives are represented
- What specific information gaps exist
- What types of sources or angles would fill those gaps

**Part 2 — Directive Generation:** Based on your gap analysis, generate research directives targeting the identified gaps.

Your response MUST follow this exact format:

<gap_analysis>
[Your detailed gap analysis here — plain text with section headings]
</gap_analysis>

```json
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
    "rationale": "Why these directives were chosen, referencing your gap analysis"
}
```

Guidelines:
- Set "research_complete" to true ONLY when existing coverage is sufficient across all dimensions. Premature completion leaves gaps that degrade report quality, but never completing wastes budget on diminishing returns. The threshold is "sufficient for a confident, well-sourced answer."
- Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences). Researchers receive this as their sole guidance — a vague directive produces a vague, unfocused research pass that wastes a full iteration budget.
- "priority": 1=critical gap, 2=important, 3=nice-to-have
- Maximum 5 directives per round
- Do NOT duplicate research already covered
- Your directives MUST directly address gaps from your analysis. Untargeted directives risk duplicating already-covered ground while leaving actual gaps unfilled.
- The gap_analysis section MUST come FIRST, before the JSON"""

    def _build_combined_think_delegate_user_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
    ) -> str:
        """Build user prompt for the combined think+delegate step."""
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

        # Prior supervisor conversation
        if state.supervision_messages:
            parts.append("## Prior Supervisor Conversation")
            parts.append(
                "Below is your conversation history from previous rounds. "
                "Reference your prior reasoning and research findings."
            )
            parts.append("")
            for msg in state.supervision_messages:
                msg_round = msg.get("round", "?")
                msg_type = msg.get("type", "unknown")
                msg_content = msg.get("content", "")
                # Sanitize tool_result content (derived from web-scraped
                # sources) to strip prompt-injection vectors before
                # interpolating into the supervision prompt.
                if msg.get("role") == "tool_result":
                    msg_content = sanitize_external_content(msg_content)
                if msg.get("role") == "assistant" and msg_type == "think":
                    parts.append(f"### [Round {msg_round}] Your Gap Analysis")
                    parts.append(msg_content)
                    parts.append("")
                elif msg.get("role") == "assistant" and msg_type == "delegation":
                    parts.append(f"### [Round {msg_round}] Your Delegation Response")
                    parts.append(msg_content)
                    parts.append("")
                elif msg.get("role") == "tool_result" and msg_type == "evidence_inventory":
                    directive_id = msg.get("directive_id", "unknown")
                    parts.append(f"### [Round {msg_round}] Evidence Inventory (directive {directive_id})")
                    parts.append(msg_content)
                    parts.append("")
                elif msg.get("role") == "tool_result":
                    directive_id = msg.get("directive_id", "unknown")
                    parts.append(f"### [Round {msg_round}] Research Findings (directive {directive_id})")
                    parts.append(msg_content)
                    parts.append("")
            parts.append("---")
            parts.append("")

        # Per-query coverage
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

        # Previously executed directives
        if state.directives:
            parts.append("## Previously Executed Directives (DO NOT repeat)")
            for d in state.directives[-10:]:
                parts.append(f"- [P{d.priority}] {d.research_topic[:120]}")
            parts.append("")

        parts.extend([
            "## Instructions",
            "1. First, write your gap analysis inside <gap_analysis> tags",
            "2. Then, if coverage is sufficient, return JSON with research_complete=true",
            "3. Otherwise, generate 1-5 detailed research directives as JSON",
            "4. Your directives must directly address gaps from your analysis",
            "",
        ])

        return "\n".join(parts)

    @staticmethod
    def _parse_combined_response(content: str) -> tuple[Optional[str], DelegationResponse]:
        """Parse a combined think+delegate response.

        Extracts the gap analysis from ``<gap_analysis>`` tags and the JSON
        directive payload from the remainder.

        Args:
            content: Raw LLM response with gap_analysis + JSON

        Returns:
            Tuple of (gap_analysis_text, DelegationResponse)

        Raises:
            ValueError: If JSON cannot be parsed
        """
        import re

        # Extract gap analysis
        gap_match = re.search(
            r"<gap_analysis>\s*(.*?)\s*</gap_analysis>",
            content,
            re.DOTALL,
        )
        think_output = gap_match.group(1).strip() if gap_match else None

        # Extract JSON (after gap_analysis section or anywhere in content)
        json_str = extract_json(content)
        if not json_str:
            raise ValueError("No JSON found in combined response")

        delegation = parse_delegation_response(json_str)
        return think_output, delegation

    @staticmethod
    def _extract_gap_analysis_section(content: str) -> Optional[str]:
        """Extract gap analysis from <gap_analysis> tags (fallback helper)."""
        import re
        match = re.search(
            r"<gap_analysis>\s*(.*?)\s*</gap_analysis>",
            content,
            re.DOTALL,
        )
        return match.group(1).strip() if match else None

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
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Propagate cancellation if any task was cancelled
        for r in gather_results:
            if isinstance(r, asyncio.CancelledError):
                raise r

        # Separate successful results from unexpected exceptions
        results: list[TopicResearchResult] = []
        for i, r in enumerate(gather_results):
            if isinstance(r, BaseException):
                logger.warning(
                    "Directive researcher unexpected exception for sub-query %s: %s. Non-fatal.",
                    directive_sub_queries[i].id,
                    r,
                )
            else:
                results.append(r)

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
    # Inline compression of directive results
    # ------------------------------------------------------------------

    async def _compress_directive_results_inline(
        self,
        state: DeepResearchState,
        directive_results: list[TopicResearchResult],
        timeout: float,
    ) -> dict[str, Any]:
        """Compress directive results inline before appending to supervision messages.

        Invokes ``_compress_single_topic_async`` for each directive result that
        has source IDs but no ``compressed_findings`` yet.  This reduces
        supervision message history growth by ~45% per round by storing
        compressed content instead of raw findings.

        Each compression is guarded by a per-result timeout.  On failure or
        timeout, the result is left without ``compressed_findings`` and the
        caller falls back to a truncated raw summary.

        Args:
            state: Current research state.
            directive_results: Results from ``_execute_directives_async``.
            timeout: Outer operation timeout (per-result timeout derived from
                config or this value).

        Returns:
            Dict with inline compression statistics.
        """
        results_to_compress = [
            r for r in directive_results
            if r.source_ids and r.compressed_findings is None
        ]

        if not results_to_compress:
            already_compressed = sum(
                1 for r in directive_results if r.compressed_findings is not None
            )
            return {
                "compressed": 0,
                "failed": 0,
                "skipped": already_compressed,
            }

        # Per-result compression timeout — same as batch compression timeout
        compression_timeout: float = getattr(
            self.config, "deep_research_compression_timeout", 120.0,
        )

        compressed_count = 0
        failed_count = 0

        async def compress_one(topic_result: TopicResearchResult) -> bool:
            try:
                # Rely on the inner timeout within _compress_single_topic_async
                # rather than double-wrapping with asyncio.wait_for (which uses
                # the same timeout value and can leave resources inconsistent
                # on outer cancellation).
                _, _, success = await self._compress_single_topic_async(
                    topic_result=topic_result,
                    state=state,
                    timeout=compression_timeout,
                )
                return success
            except asyncio.TimeoutError:
                logger.warning(
                    "Inline compression timed out for directive %s",
                    topic_result.sub_query_id,
                )
                return False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Inline compression failed for directive %s: %s",
                    topic_result.sub_query_id,
                    exc,
                )
                return False

        tasks = [compress_one(r) for r in results_to_compress]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Propagate cancellation if any task was cancelled
        for r in gather_results:
            if isinstance(r, asyncio.CancelledError):
                raise r

        for result in gather_results:
            if isinstance(result, BaseException) or not result:
                failed_count += 1
            else:
                compressed_count += 1

        self._write_audit_event(
            state,
            "inline_directive_compression",
            data={
                "compressed": compressed_count,
                "failed": failed_count,
                "total": len(results_to_compress),
            },
        )

        logger.info(
            "Inline directive compression: %d/%d compressed, %d failed",
            compressed_count,
            len(results_to_compress),
            failed_count,
        )

        return {
            "compressed": compressed_count,
            "failed": failed_count,
            "skipped": len(directive_results) - len(results_to_compress),
        }

    @staticmethod
    def _build_directive_fallback_summary(
        topic_result: TopicResearchResult,
        state: DeepResearchState,
        max_chars: int = 800,
    ) -> Optional[str]:
        """Build a truncated fallback summary when inline compression fails.

        Used as the ``tool_result`` content in supervision messages when
        ``_compress_single_topic_async`` fails or times out.  Prefers
        ``per_topic_summary`` when available; otherwise builds a brief
        summary from source titles and content snippets.

        Args:
            topic_result: The directive's topic research result.
            state: Current research state (for source lookup).
            max_chars: Maximum character length for the fallback (default 800).

        Returns:
            Truncated summary string, or None if no content is available.
        """
        # Prefer per_topic_summary if available
        if topic_result.per_topic_summary:
            summary = topic_result.per_topic_summary
            if len(summary) > max_chars:
                return summary[:max_chars] + "..."
            return summary

        # Build from source content
        topic_sources = [
            s for s in state.sources if s.id in topic_result.source_ids
        ]
        if not topic_sources:
            return None

        parts: list[str] = []
        remaining = max_chars
        for src in topic_sources:
            content = sanitize_external_content(src.content or src.snippet or "")
            title = sanitize_external_content(src.title or "Untitled")
            if content:
                entry = f"- {title}: {content[:200]}"
            else:
                entry = f"- {title}"
            if remaining < len(entry):
                break
            parts.append(entry)
            remaining -= len(entry)

        return "\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Evidence inventory for supervisor context preservation
    # ------------------------------------------------------------------

    # Maximum character length for a single evidence inventory message.
    # Keeps token overhead bounded (~125 tokens at 4 chars/token).
    _EVIDENCE_INVENTORY_MAX_CHARS: int = 500

    @staticmethod
    def _build_evidence_inventory(
        topic_result: TopicResearchResult,
        state: DeepResearchState,
        max_chars: int = _EVIDENCE_INVENTORY_MAX_CHARS,
    ) -> Optional[str]:
        """Build a compact evidence inventory from a directive result.

        Produces a structured, short summary listing sources found (URL +
        title + topic coverage), key data point count, and topics addressed.
        This gives the supervisor specific evidence to reason about without
        the full token cost of raw notes or compressed findings.

        Matches the ODR pattern where the supervisor sees both compressed
        notes and a separate evidence reference for each researcher's output.

        Args:
            topic_result: The directive's topic research result (with
                ``source_ids`` and optionally ``raw_notes``).
            state: Current research state (for source metadata lookup).
            max_chars: Maximum character length for the inventory
                (default 500).

        Returns:
            Compact inventory string, or ``None`` if no evidence exists.
        """
        if not topic_result.source_ids and not topic_result.raw_notes:
            return None

        # Gather source metadata
        source_map = {s.id: s for s in state.sources}
        topic_sources = [
            source_map[sid] for sid in topic_result.source_ids
            if sid in source_map
        ]

        if not topic_sources and not topic_result.raw_notes:
            return None

        parts: list[str] = []

        # Source summary line
        unique_domains: set[str] = set()
        for src in topic_sources:
            if src.url:
                try:
                    domain = urlparse(src.url).netloc
                    if domain:
                        unique_domains.add(domain)
                except Exception:
                    pass

        parts.append(
            f"Sources: {len(topic_sources)} found, "
            f"{len(unique_domains)} unique domain{'s' if len(unique_domains) != 1 else ''}"
        )

        # Per-source entries (compact: number + title + domain)
        remaining = max_chars - len(parts[0]) - 2  # reserve for newlines
        for idx, src in enumerate(topic_sources, 1):
            domain = ""
            if src.url:
                try:
                    domain = urlparse(src.url).netloc
                except Exception:
                    pass
            title = sanitize_external_content((src.title or "Untitled"))[:60]
            entry = f"- [{idx}] \"{title}\""
            if domain:
                entry += f" ({domain})"
            if len(entry) + 1 > remaining:
                break
            parts.append(entry)
            remaining -= len(entry) + 1

        # Key findings from supervisor summary (structured for gap analysis)
        if topic_result.supervisor_summary:
            # Truncate to fit within remaining budget
            brief = topic_result.supervisor_summary
            label = "Key findings: "
            max_brief = remaining - len(label) - 1
            if max_brief > 20:
                if len(brief) > max_brief:
                    brief = brief[:max_brief - 3] + "..."
                findings_line = f"{label}{brief}"
                parts.append(findings_line)
                remaining -= len(findings_line) + 1

        # Data point estimate from raw notes (count paragraphs as proxy)
        if topic_result.raw_notes and remaining > 30:
            # Count non-empty lines as a rough data-point proxy
            lines = [
                ln for ln in topic_result.raw_notes.split("\n")
                if ln.strip()
            ]
            data_points = min(len(lines), 999)
            dp_line = f"Key data points: ~{data_points} extracted"
            if len(dp_line) + 1 <= remaining:
                parts.append(dp_line)
                remaining -= len(dp_line) + 1

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result

    # ------------------------------------------------------------------
    # Query complexity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_query_complexity(
        state: DeepResearchState,
    ) -> str:
        """Classify the original query's complexity for directive scaling.

        Uses heuristics based on sub-query count and research brief length
        to produce a simple/moderate/complex label that guides the supervisor
        in calibrating how many directives to generate.

        Args:
            state: Current research state (uses sub_queries, research_brief,
                   original_query)

        Returns:
            One of ``"simple"``, ``"moderate"``, or ``"complex"``
        """
        sub_query_count = len(state.sub_queries)
        brief = state.research_brief or state.original_query
        brief_word_count = len(brief.split())

        # High sub-query count or long brief → complex
        if sub_query_count >= 5 or brief_word_count >= 200:
            return "complex"

        # Moderate indicators
        if sub_query_count >= 3 or brief_word_count >= 80:
            return "moderate"

        return "simple"

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
- "priority": 1=critical gap (blocks report quality), 2=important (improves comprehensiveness), 3=nice-to-have. Priority determines execution order when budget is limited — critical gaps are addressed first because they block the report from being useful; nice-to-have gaps are only pursued if budget remains.
- Do NOT duplicate research already covered — target SPECIFIC gaps
- Directives should be complementary, not overlapping — each covers a different dimension
- Do NOT use acronyms or abbreviations in directive text — spell out all terms so researchers search for the correct concepts. Acronyms may be ambiguous (e.g., "ML" could mean machine learning or markup language) and produce irrelevant search results.

Directive Count Scaling:
- Simple factual gaps (single missing fact or stat): 1-2 directives maximum
- Comparison gaps (need data on specific compared elements): 1 directive per element needing more research
- Complex multi-dimensional gaps (multiple interrelated areas uncovered): 3-5 directives targeting distinct dimensions
- BIAS toward fewer, more focused directives — a single well-scoped directive beats three vague ones. Each directive spawns a full researcher agent with its own search budget; fewer, focused directives concentrate budget on the actual gaps, while many vague ones spread budget thin and produce overlapping, shallow results.
- Maximum 5 directives per round regardless of complexity. Each directive consumes a researcher agent's full budget — more than 5 per round risks exceeding the session's total budget and hitting diminishing returns before the next supervision assessment.

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

        # Complexity signal for directive scaling
        complexity = self._classify_query_complexity(state)
        complexity_guidance = {
            "simple": "This is a **simple** query — target 1-2 focused directives for remaining gaps.",
            "moderate": "This is a **moderate** complexity query — target 2-3 directives for remaining gaps.",
            "complex": "This is a **complex** multi-dimensional query — target 3-5 directives for remaining gaps.",
        }

        parts.extend([
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Supervision round: {state.supervision_round + 1}/{state.max_supervision_rounds}",
            f"- Completed sub-queries: {len(state.completed_sub_queries())}",
            f"- Total sources: {len(state.sources)}",
            f"- Query complexity: **{complexity}**",
            "",
            complexity_guidance[complexity],
            "",
        ])

        # Prior supervisor conversation (accumulated across rounds)
        if state.supervision_messages:
            parts.append("## Prior Supervisor Conversation")
            parts.append(
                "Below is your conversation history from previous rounds. "
                "Reference your prior reasoning and the research findings "
                "to avoid re-delegating already-covered topics."
            )
            parts.append("")
            for msg in state.supervision_messages:
                msg_round = msg.get("round", "?")
                msg_type = msg.get("type", "unknown")
                msg_content = msg.get("content", "")
                # Sanitize tool_result content (derived from web-scraped
                # sources) to strip prompt-injection vectors before
                # interpolating into the supervision prompt.
                if msg.get("role") == "tool_result":
                    msg_content = sanitize_external_content(msg_content)
                if msg.get("role") == "assistant" and msg_type == "think":
                    parts.append(f"### [Round {msg_round}] Your Gap Analysis")
                    parts.append(msg_content)
                    parts.append("")
                elif msg.get("role") == "assistant" and msg_type == "delegation":
                    parts.append(f"### [Round {msg_round}] Your Delegation Response")
                    parts.append(msg_content)
                    parts.append("")
                elif msg.get("role") == "tool_result" and msg_type == "evidence_inventory":
                    directive_id = msg.get("directive_id", "unknown")
                    parts.append(f"### [Round {msg_round}] Evidence Inventory (directive {directive_id})")
                    parts.append(msg_content)
                    parts.append("")
                elif msg.get("role") == "tool_result":
                    directive_id = msg.get("directive_id", "unknown")
                    parts.append(f"### [Round {msg_round}] Research Findings (directive {directive_id})")
                    parts.append(msg_content)
                    parts.append("")
            parts.append("---")
            parts.append("")

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

        # Gap analysis reference — the full think output is already in the
        # "Prior Supervisor Conversation" section above (Phase 6: think flows
        # through message history).  We include a lightweight instruction
        # rather than duplicating the full analysis text.
        if think_output:
            if state.supervision_messages:
                parts.extend([
                    "## Gap Analysis",
                    "",
                    "Your gap analysis from this round is in the conversation "
                    "history above. Generate research directives that DIRECTLY "
                    "address the gaps you identified.",
                    "Each directive should target a specific gap with a detailed "
                    "research plan.",
                    "",
                ])
            else:
                # Fallback: no conversation history (shouldn't happen, but safe)
                parts.extend([
                    "## Gap Analysis",
                    "",
                    "<gap_analysis>",
                    think_output.strip(),
                    "</gap_analysis>",
                    "",
                    "Generate research directives that DIRECTLY address the gaps "
                    "identified above.",
                    "Each directive should target a specific gap with a detailed "
                    "research plan.",
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

    def _apply_directive_caps(
        self,
        directives: list[ResearchDirective],
        state: DeepResearchState,
    ) -> list[ResearchDirective]:
        """Apply business rules to structured-parsed directives.

        Caps directive count, filters empty topics, and stamps
        ``supervision_round`` on each directive. This is the structured-output
        equivalent of the filtering logic in ``_parse_delegation_response()``.

        Args:
            directives: Raw directives from DelegationResponse schema
            state: Current research state

        Returns:
            Capped and validated directive list
        """
        max_units = getattr(
            self.config, "deep_research_max_concurrent_research_units", 5,
        )
        cap = min(max_units, _MAX_DIRECTIVES_PER_ROUND)

        result: list[ResearchDirective] = []
        for d in directives[:cap]:
            if not d.research_topic.strip():
                continue
            d.supervision_round = state.supervision_round
            result.append(d)
        return result

    def _parse_delegation_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> tuple[list[ResearchDirective], bool]:
        """Parse LLM response into research directives (legacy fallback).

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
            "priority": 1  // 1=critical, 2=important, 3=supplementary
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
- Directives should be SPECIFIC enough to yield targeted search results
- Directives must cover DISTINCT aspects — no two should investigate substantially the same ground
- Do NOT use acronyms or abbreviations in directive text — spell out all terms so researchers search for the correct concepts

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
                "research assignment for a specialized researcher — sub-agents cannot "
                "see other agents' work, so every directive must include full context.",
                "",
            ])

        parts.extend([
            "## Instructions",
            "1. Decompose the research query into 2-5 focused research directives",
            "2. Each directive should target a distinct aspect of the query",
            "3. Each directive should be specific enough to yield targeted results",
            "4. Prioritize: 1=critical to the core question, 2=important for comprehensiveness, 3=supplementary",
            "",
            "Return your response as JSON.",
        ])

        return "\n".join(parts)

    # ==================================================================
    # First-round decompose → critique → revise pipeline
    # ==================================================================

    async def _first_round_decompose_critique_revise(
        self,
        state: DeepResearchState,
        think_output: Optional[str],
        provider_id: Optional[str],
        timeout: float,
    ) -> tuple[list[ResearchDirective], bool, Optional[str]]:
        """Three-call pipeline for first-round query decomposition.

        Replaces the single-call first-round delegation with a pipeline that
        separates generation, critique, and revision into distinct LLM calls
        for higher-quality directives:

        1. **Generate** — decompose the query into initial directives (existing
           first-round prompts).
        2. **Critique** — evaluate the initial directives for redundancy,
           coverage gaps, proportionality, and specificity issues.
        3. **Revise** — apply the critique to produce the final directive set.
           Skipped if the critique finds no issues.

        Args:
            state: Current research state
            think_output: Decomposition strategy from think step
            provider_id: LLM provider to use
            timeout: Request timeout

        Returns:
            Tuple of (directives list, research_complete flag, raw content)
        """
        effective_provider = provider_id or state.supervision_provider

        # --- Call 1: Generate initial directives ---
        self._check_cancellation(state)
        gen_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate_generate",
            system_prompt=self._build_first_round_delegation_system_prompt(),
            user_prompt=self._build_first_round_delegation_user_prompt(
                state, think_output,
            ),
            provider_id=effective_provider,
            model=state.supervision_model,
            temperature=0.3,
            timeout=timeout,
            parse_fn=parse_delegation_response,
            role="delegation",
        )

        if isinstance(gen_result, WorkflowResult):
            logger.warning(
                "First-round generate call failed: %s", gen_result.error,
            )
            return [], False, None

        # Extract initial directives
        if gen_result.parsed is not None:
            gen_delegation: DelegationResponse = gen_result.parsed
            initial_directives = self._apply_directive_caps(
                gen_delegation.directives, state,
            )
            research_complete = gen_delegation.research_complete
        else:
            logger.warning(
                "First-round generate parse failed, falling back to legacy parser",
            )
            initial_directives, research_complete = self._parse_delegation_response(
                gen_result.result.content, state,
            )

        initial_count = len(initial_directives)

        self._write_audit_event(
            state,
            "first_round_generate",
            data={
                "provider_id": gen_result.result.provider_id,
                "model_used": gen_result.result.model_used,
                "tokens_used": gen_result.result.tokens_used,
                "directive_count": initial_count,
                "research_complete": research_complete,
                "directive_topics": [
                    d.research_topic[:100] for d in initial_directives
                ],
            },
        )

        # If generation signalled research_complete or produced no directives,
        # skip critique/revision — there's nothing to refine.
        if research_complete or not initial_directives:
            self._write_audit_event(
                state,
                "first_round_decomposition",
                data={
                    "initial_directive_count": initial_count,
                    "final_directive_count": initial_count,
                    "critique_triggered_revision": False,
                    "skip_reason": (
                        "research_complete" if research_complete
                        else "no_directives"
                    ),
                },
            )
            return initial_directives, research_complete, gen_result.result.content

        # --- Call 2: Critique the initial directives ---
        self._check_cancellation(state)
        directives_json = json.dumps(
            [
                {
                    "research_topic": d.research_topic,
                    "perspective": d.perspective,
                    "evidence_needed": d.evidence_needed,
                    "priority": d.priority,
                }
                for d in initial_directives
            ],
            indent=2,
        )

        critique_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate_critique",
            system_prompt=self._build_critique_system_prompt(),
            user_prompt=self._build_critique_user_prompt(
                state, directives_json,
            ),
            provider_id=effective_provider,
            model=state.supervision_model,
            temperature=0.2,
            timeout=getattr(
                self.config, "deep_research_reflection_timeout", 60.0,
            ),
            role="reflection",
        )

        if isinstance(critique_result, WorkflowResult):
            logger.warning(
                "First-round critique call failed: %s. Using initial directives.",
                critique_result.error,
            )
            self._write_audit_event(
                state,
                "first_round_decomposition",
                data={
                    "initial_directive_count": initial_count,
                    "final_directive_count": initial_count,
                    "critique_triggered_revision": False,
                    "skip_reason": "critique_failed",
                },
            )
            return initial_directives, research_complete, gen_result.result.content

        critique_text = critique_result.result.content or ""

        # Detect whether the critique flagged any issues worth revising.
        needs_revision = self._critique_has_issues(critique_text)

        self._write_audit_event(
            state,
            "first_round_critique",
            data={
                "provider_id": critique_result.result.provider_id,
                "model_used": critique_result.result.model_used,
                "tokens_used": critique_result.result.tokens_used,
                "needs_revision": needs_revision,
                "critique_length": len(critique_text),
            },
        )

        # --- Call 3: Revise (skip if critique found no issues) ---
        if not needs_revision:
            logger.info(
                "First-round critique found no issues, using initial directives",
            )
            self._write_audit_event(
                state,
                "first_round_decomposition",
                data={
                    "initial_directive_count": initial_count,
                    "final_directive_count": initial_count,
                    "critique_triggered_revision": False,
                },
            )
            return initial_directives, research_complete, gen_result.result.content

        self._check_cancellation(state)
        revise_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate_revise",
            system_prompt=self._build_revision_system_prompt(),
            user_prompt=self._build_revision_user_prompt(
                state, directives_json, critique_text,
            ),
            provider_id=effective_provider,
            model=state.supervision_model,
            temperature=0.3,
            timeout=timeout,
            parse_fn=parse_delegation_response,
            role="delegation",
        )

        if isinstance(revise_result, WorkflowResult):
            logger.warning(
                "First-round revision call failed: %s. Using initial directives.",
                revise_result.error,
            )
            self._write_audit_event(
                state,
                "first_round_decomposition",
                data={
                    "initial_directive_count": initial_count,
                    "final_directive_count": initial_count,
                    "critique_triggered_revision": True,
                    "revision_failed": True,
                },
            )
            return initial_directives, research_complete, gen_result.result.content

        # Extract revised directives
        if revise_result.parsed is not None:
            rev_delegation: DelegationResponse = revise_result.parsed
            final_directives = self._apply_directive_caps(
                rev_delegation.directives, state,
            )
            research_complete = rev_delegation.research_complete
        else:
            logger.warning(
                "Revision parse failed, falling back to legacy parser",
            )
            final_directives, research_complete = self._parse_delegation_response(
                revise_result.result.content, state,
            )

        final_count = len(final_directives)

        self._write_audit_event(
            state,
            "first_round_revise",
            data={
                "provider_id": revise_result.result.provider_id,
                "model_used": revise_result.result.model_used,
                "tokens_used": revise_result.result.tokens_used,
                "directive_count": final_count,
                "directive_topics": [
                    d.research_topic[:100] for d in final_directives
                ],
            },
        )

        self._write_audit_event(
            state,
            "first_round_decomposition",
            data={
                "initial_directive_count": initial_count,
                "final_directive_count": final_count,
                "critique_triggered_revision": True,
                "directives_delta": final_count - initial_count,
            },
        )

        logger.info(
            "First-round decompose→critique→revise: %d → %d directives",
            initial_count,
            final_count,
        )

        # Return the revision's raw content for message accumulation
        raw_content = revise_result.result.content
        return final_directives, research_complete, raw_content

    def _build_critique_system_prompt(self) -> str:
        """Build system prompt for the critique call (call 2 of 3).

        Instructs the LLM to evaluate a set of research directives against
        four quality criteria without revising them.
        """
        return """You are a research quality reviewer. You will receive a set of research directives generated for a query. Your task is to critique them — identify issues but do NOT revise the directives yourself.

Evaluate the directives against these four criteria:

1. **Redundancy**: Are any directives investigating the same topic from the same angle? If so, identify which ones overlap and should be merged.
2. **Coverage**: Is there a major dimension of the query that no directive addresses? If so, identify what perspective or facet is missing.
3. **Proportionality**: Given the complexity of the query, is the number of directives appropriate? A simple factual question needs 1-2 directives, not 4-5. A complex multi-dimensional topic warrants 3-5.
4. **Specificity**: Are all directives specific enough to yield targeted search results? Identify any that are too broad or vague.

For each criterion, state either:
- "PASS" — no issues found
- "ISSUE: <description>" — describe the specific problem

End your response with a summary line:
- "VERDICT: NO_ISSUES" — if all four criteria pass
- "VERDICT: REVISION_NEEDED" — if any criterion has issues

Be concise and specific. Focus on actionable feedback."""

    def _build_critique_user_prompt(
        self,
        state: DeepResearchState,
        directives_json: str,
    ) -> str:
        """Build user prompt for the critique call.

        Args:
            state: Current research state (for the original query)
            directives_json: JSON string of the initial directives
        """
        return "\n".join([
            f"# Original Research Query\n{state.original_query}",
            "",
            "# Directives to Critique",
            directives_json,
            "",
            "Evaluate the directives above against the four criteria "
            "(redundancy, coverage, proportionality, specificity).",
        ])

    def _build_revision_system_prompt(self) -> str:
        """Build system prompt for the revision call (call 3 of 3).

        Instructs the LLM to revise directives based on critique feedback.
        """
        return """You are a research lead revising a set of research directives based on critique feedback. Apply the critique to produce an improved directive set.

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
    "rationale": "How the critique was applied to improve the directives"
}

Revision Guidelines:
- MERGE directives flagged as redundant into single stronger directives
- ADD a directive for any missing coverage identified in the critique
- REMOVE excess directives if proportionality issues were flagged
- SHARPEN any directives flagged as too broad or vague
- Keep directives that were not flagged unchanged
- Maintain 2-5 directives total
- Each "research_topic" should be a detailed paragraph (2-4 sentences)

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_revision_user_prompt(
        self,
        state: DeepResearchState,
        directives_json: str,
        critique_text: str,
    ) -> str:
        """Build user prompt for the revision call.

        Args:
            state: Current research state (for the original query)
            directives_json: JSON string of the initial directives
            critique_text: Critique feedback from call 2
        """
        return "\n".join([
            f"# Original Research Query\n{state.original_query}",
            "",
            "# Current Directives",
            directives_json,
            "",
            "# Critique Feedback",
            critique_text,
            "",
            "Revise the directives based on the critique above. "
            "Return the improved directive set as JSON.",
        ])

    # Pre-compiled patterns for verdict/issue parsing (used by _critique_has_issues)
    _VERDICT_NO_ISSUES_RE = re.compile(r"VERDICT\s*:\s*NO[_\s]?ISSUES", re.IGNORECASE)
    _VERDICT_REVISION_RE = re.compile(r"VERDICT\s*:\s*REVISION[_\s]?NEEDED", re.IGNORECASE)
    # Match "ISSUE:" as a structured marker — either at line start, after numbering,
    # or after a label prefix (e.g. "Redundancy: ISSUE:"). Avoids false positives
    # on conversational uses like "this is not an issue" by requiring the colon.
    _ISSUE_MARKER_RE = re.compile(r"(?:^|\.\s+|:\s*)ISSUE\s*:", re.IGNORECASE | re.MULTILINE)

    @staticmethod
    def _critique_has_issues(critique_text: str) -> bool:
        """Check whether the critique indicates issues that need revision.

        Looks for the ``VERDICT:`` line with flexible whitespace/formatting.
        Falls back to checking for ``ISSUE:`` markers at line starts if no
        verdict is found.

        Args:
            critique_text: Raw text from the critique LLM call

        Returns:
            True if revision is needed, False if all criteria passed
        """
        cls = SupervisionPhaseMixin
        if cls._VERDICT_NO_ISSUES_RE.search(critique_text):
            return False
        if cls._VERDICT_REVISION_RE.search(critique_text):
            return True
        # Fallback: check for ISSUE markers at line starts to reduce false positives
        return bool(cls._ISSUE_MARKER_RE.search(critique_text))

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
        compatibility.

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
                    "Supervision heuristic: confidence %.2f >= threshold "
                    "(round %d), advancing to analysis",
                    heuristic.get("confidence", 0.0),
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
                # Prefer supervisor_summary (structured for gap analysis) over
                # raw compressed_findings truncation when available.
                if topic_result.supervisor_summary:
                    compressed_findings_excerpt = topic_result.supervisor_summary
                elif topic_result.compressed_findings:
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

    def _store_coverage_snapshot(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
    ) -> None:
        """Store a coverage snapshot for the current supervision round.

        Snapshots are keyed by round number in ``state.metadata["coverage_snapshots"]``
        and used by ``_compute_coverage_delta`` to produce round-over-round deltas.

        Each snapshot entry stores the lightweight fields needed for delta
        comparison: source_count, unique_domains, and status.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage from ``_build_per_query_coverage``
        """
        snapshots = state.metadata.setdefault("coverage_snapshots", {})
        snapshot: dict[str, dict[str, Any]] = {}
        for entry in coverage_data:
            snapshot[entry["sub_query_id"]] = {
                "query": entry["query"],
                "source_count": entry["source_count"],
                "unique_domains": entry["unique_domains"],
                "status": entry["status"],
            }
        # Store with string key (JSON-safe)
        snapshots[str(state.supervision_round)] = snapshot

    def _compute_coverage_delta(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        min_sources: int = 3,
    ) -> Optional[str]:
        """Compute a coverage delta between the current and previous supervision round.

        Compares per-query source counts, domain counts, and status against the
        snapshot from the previous round.  Produces a compact summary like::

            Coverage delta (round 0 → 1):
            - query_1: +2 sources, +1 domain (now: 4 sources, 3 domains) — SUFFICIENT
            - query_2: +0 sources — STILL INSUFFICIENT
            - query_3 [NEW]: 1 source from this round's directives

        Returns ``None`` if there is no previous snapshot (round 0) or if
        coverage_snapshots metadata is missing.

        Args:
            state: Current research state
            coverage_data: Current per-sub-query coverage
            min_sources: Minimum sources per query for "SUFFICIENT" label

        Returns:
            Compact delta summary string, or ``None`` if no previous snapshot exists
        """
        snapshots = state.metadata.get("coverage_snapshots", {})
        prev_round = state.supervision_round - 1
        prev_snapshot = snapshots.get(str(prev_round))
        if prev_snapshot is None:
            return None

        # Build current lookup
        current_by_id: dict[str, dict[str, Any]] = {}
        for entry in coverage_data:
            current_by_id[entry["sub_query_id"]] = entry

        lines: list[str] = [
            f"Coverage delta (round {prev_round} → {state.supervision_round}):",
        ]

        # Track IDs we've seen to detect new queries
        prev_ids = set(prev_snapshot.keys())
        current_ids = set(current_by_id.keys())

        # Process queries that existed in previous round
        for sq_id in sorted(prev_ids & current_ids):
            prev = prev_snapshot[sq_id]
            curr = current_by_id[sq_id]
            src_delta = curr["source_count"] - prev["source_count"]
            dom_delta = curr["unique_domains"] - prev["unique_domains"]

            # Determine sufficiency label
            if curr["source_count"] >= min_sources:
                if prev["source_count"] < min_sources:
                    status_label = "NEWLY SUFFICIENT"
                else:
                    status_label = "SUFFICIENT"
            else:
                status_label = "STILL INSUFFICIENT"

            src_sign = f"+{src_delta}" if src_delta >= 0 else str(src_delta)
            dom_sign = f"+{dom_delta}" if dom_delta >= 0 else str(dom_delta)

            query_text = curr.get("query", prev.get("query", sq_id))[:80]
            lines.append(
                f"- {query_text}: {src_sign} sources, {dom_sign} domains "
                f"(now: {curr['source_count']} sources, {curr['unique_domains']} domains) "
                f"— {status_label}"
            )

        # Process new queries (not in previous round)
        for sq_id in sorted(current_ids - prev_ids):
            curr = current_by_id[sq_id]
            query_text = curr.get("query", sq_id)[:80]
            lines.append(
                f"- {query_text} [NEW]: "
                f"{curr['source_count']} sources, {curr['unique_domains']} domains"
            )

        # Process removed queries (in previous but not current — rare)
        for sq_id in sorted(prev_ids - current_ids):
            prev = prev_snapshot[sq_id]
            query_text = prev.get("query", sq_id)[:80]
            lines.append(f"- {query_text} [REMOVED]")

        return "\n".join(lines)

    def _build_think_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        coverage_delta: Optional[str] = None,
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

        When a ``coverage_delta`` is provided (rounds > 0), it is injected
        before the per-query coverage section so the LLM can focus its analysis
        on what actually changed since the last round.

        Args:
            state: Current research state
            coverage_data: Per-sub-query coverage from _build_per_query_coverage
            coverage_delta: Optional delta summary from ``_compute_coverage_delta``

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

        # Coverage delta (injected before full coverage for focus)
        if coverage_delta:
            parts.append("## What Changed Since Last Round\n")
            parts.append(coverage_delta)
            parts.append("")
            parts.append(
                "*Focus your analysis on the changes above. Queries marked "
                "STILL INSUFFICIENT or NEW deserve the most attention.*"
            )
            parts.append("")

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
        """Assess coverage using multi-dimensional confidence scoring.

        Computes a confidence score from three dimensions:

        - **Source adequacy**: average of ``min(1.0, sources / min_sources)``
          across all completed sub-queries.
        - **Domain diversity**: ``unique_domains / (query_count * 2)`` capped
          at 1.0 — rewards breadth of sourcing.
        - **Query completion rate**: ``completed / total`` sub-queries.

        The overall confidence is a weighted mean (configurable via
        ``deep_research_coverage_confidence_weights``).  When
        ``confidence >= threshold`` (default 0.75), the heuristic declares
        coverage sufficient and sets ``should_continue_gathering=False``.

        Args:
            state: Current research state
            min_sources: Minimum sources per sub-query for "sufficient" coverage

        Returns:
            Dict with coverage assessment, confidence breakdown, and
            should_continue_gathering flag
        """
        completed = state.completed_sub_queries()
        total_queries = len(state.sub_queries)

        if not completed:
            return {
                "overall_coverage": "insufficient",
                "should_continue_gathering": False,
                "queries_assessed": 0,
                "queries_sufficient": 0,
                "confidence": 0.0,
                "confidence_dimensions": {
                    "source_adequacy": 0.0,
                    "domain_diversity": 0.0,
                    "query_completion_rate": 0.0,
                },
                "dominant_factors": [],
                "weak_factors": ["source_adequacy", "domain_diversity", "query_completion_rate"],
            }

        # --- Dimension 1: Source adequacy ---
        source_ratios: list[float] = []
        sufficient_count = 0
        for sq in completed:
            sq_sources = [s for s in state.sources if s.sub_query_id == sq.id]
            count = len(sq_sources)
            ratio = min(1.0, count / min_sources) if min_sources > 0 else 1.0
            source_ratios.append(ratio)
            if count >= min_sources:
                sufficient_count += 1
        source_adequacy = sum(source_ratios) / len(source_ratios)

        # --- Dimension 2: Domain diversity ---
        all_domains: set[str] = set()
        for s in state.sources:
            if s.url:
                try:
                    parsed = urlparse(s.url)
                    if parsed.netloc:
                        all_domains.add(parsed.netloc)
                except Exception:
                    pass
        query_count = len(completed)
        domain_diversity = min(1.0, len(all_domains) / (query_count * 2)) if query_count > 0 else 0.0

        # --- Dimension 3: Query completion rate ---
        query_completion_rate = len(completed) / total_queries if total_queries > 0 else 0.0

        # --- Weighted confidence ---
        weights = getattr(
            self.config,
            "deep_research_coverage_confidence_weights",
            None,
        ) or {"source_adequacy": 0.5, "domain_diversity": 0.2, "query_completion_rate": 0.3}
        total_weight = sum(weights.values())
        dimensions = {
            "source_adequacy": source_adequacy,
            "domain_diversity": domain_diversity,
            "query_completion_rate": query_completion_rate,
        }
        confidence = sum(
            dimensions[k] * weights.get(k, 0.0) for k in dimensions
        ) / total_weight if total_weight > 0 else 0.0

        # --- Factor classification ---
        strong_threshold = 0.7
        weak_threshold = 0.5
        dominant_factors = [k for k, v in dimensions.items() if v >= strong_threshold]
        weak_factors = [k for k, v in dimensions.items() if v < weak_threshold]

        # --- Overall coverage label (backward-compatible) ---
        if sufficient_count == query_count:
            overall = "sufficient"
        elif sufficient_count > 0:
            overall = "partial"
        else:
            overall = "insufficient"

        # --- Confidence-based decision ---
        threshold = getattr(
            self.config,
            "deep_research_coverage_confidence_threshold",
            0.75,
        )
        should_continue = confidence < threshold

        return {
            "overall_coverage": overall,
            "should_continue_gathering": should_continue,
            "queries_assessed": query_count,
            "queries_sufficient": sufficient_count,
            "confidence": round(confidence, 4),
            "confidence_threshold": threshold,
            "confidence_dimensions": dimensions,
            "dominant_factors": dominant_factors,
            "weak_factors": weak_factors,
        }
