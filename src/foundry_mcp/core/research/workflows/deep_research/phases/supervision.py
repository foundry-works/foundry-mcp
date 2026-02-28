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

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory
from urllib.parse import urlparse

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    DelegationResponse,
    ResearchDirective,
    TopicResearchResult,
    parse_delegation_response,
)
from foundry_mcp.core.research.models.sources import SubQuery
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research._json_parsing import (
    extract_json,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_llm_call,
    execute_structured_llm_call,
    finalize_phase,
    truncate_supervision_messages,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
    assess_coverage_heuristic,
    build_per_query_coverage,
    compute_coverage_delta,
    critique_has_issues,
    store_coverage_snapshot,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision_prompts import (
    build_combined_think_delegate_system_prompt,
    build_combined_think_delegate_user_prompt,
    build_critique_system_prompt,
    build_critique_user_prompt,
    build_delegation_system_prompt,
    build_delegation_user_prompt,
    build_first_round_delegation_system_prompt,
    build_first_round_delegation_user_prompt,
    build_first_round_think_prompt,
    build_first_round_think_system_prompt,
    build_revision_system_prompt,
    build_revision_user_prompt,
    build_think_prompt,
    build_think_system_prompt,
    classify_query_complexity,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import _normalize_title

logger = logging.getLogger(__name__)

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

# Cap supervision_history entries to prevent unbounded state growth.
# Consistent with _MAX_STORED_DIRECTIVES — only the most recent rounds matter.
_MAX_SUPERVISION_HISTORY_ENTRIES = 10

# Truncate think_output / post_execution_think before storing in history.
# Full text is available in audit events; the history only needs summaries.
_MAX_THINK_OUTPUT_STORED_CHARS = 2000


def _trim_supervision_history(state: Any) -> None:
    """Cap supervision_history to the most recent entries and truncate think fields."""
    history = state.metadata.get("supervision_history")
    if not history:
        return
    # Trim to most recent entries
    if len(history) > _MAX_SUPERVISION_HISTORY_ENTRIES:
        state.metadata["supervision_history"] = history[-_MAX_SUPERVISION_HISTORY_ENTRIES:]


class SupervisionPhaseMixin:
    """Supervision phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    - _execute_topic_research_async() (from TopicResearchMixin, for delegation)
    - _get_search_provider() (from GatheringPhaseMixin, for delegation)

    See ``DeepResearchWorkflowProtocol`` in ``_protocols.py`` for the
    full structural contract.
    """

    config: ResearchConfig
    memory: ResearchMemory

    # Stubs for Pyright — canonical signatures live in _protocols.py
    if TYPE_CHECKING:

        def _write_audit_event(
            self,
            state: DeepResearchState | None,
            event_name: str,
            *,
            data: dict[str, Any] | None = ...,
            level: str = ...,
        ) -> None: ...
        def _check_cancellation(self, state: DeepResearchState) -> None: ...
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
            state,
            provider_id,
            timeout,
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
        """Execute supervision via think→delegate→execute→assess loop."""
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

        total_directives_executed = 0
        total_new_sources = 0
        wall_clock_start = time.monotonic()
        wall_clock_limit: float = getattr(
            self.config,
            "deep_research_supervision_wall_clock_timeout",
            1800.0,
        )

        while state.supervision_round < state.max_supervision_rounds:
            self._check_cancellation(state)

            if self._should_exit_wall_clock(state, wall_clock_start, wall_clock_limit):
                break

            logger.info(
                "Supervision delegation round %d/%d: %d completed sub-queries, %d sources",
                state.supervision_round + 1,
                state.max_supervision_rounds,
                len(state.completed_sub_queries()),
                len(state.sources),
            )

            coverage_data, coverage_delta = self._prepare_round_coverage(
                state,
                min_sources,
            )

            # Heuristic early-exit (round > 0)
            should_exit, heuristic_data = self._should_exit_heuristic(state, min_sources)
            if should_exit:
                logger.info(
                    "Supervision delegation: confidence %.2f >= threshold at round %d, advancing",
                    heuristic_data.get("confidence", 0.0),
                    state.supervision_round,
                )
                self._record_supervision_exit(
                    state,
                    method="delegation_heuristic",
                    overall_coverage="sufficient",
                    audit_data={
                        "reason": "heuristic_sufficient",
                        "model": "delegation",
                        "supervision_round": state.supervision_round,
                        "coverage_summary": heuristic_data,
                    },
                )
                break

            # Think + Delegate
            think_output, directives, research_complete = await self._run_think_delegate_step(
                state,
                coverage_data,
                coverage_delta,
                provider_id,
                timeout,
            )

            if research_complete:
                logger.info(
                    "Supervision delegation: ResearchComplete signal at round %d",
                    state.supervision_round,
                )
                self._record_supervision_exit(
                    state,
                    method="delegation_complete",
                    overall_coverage="sufficient",
                    think_output=think_output,
                )
                break

            if not directives:
                logger.info(
                    "Supervision delegation: no directives at round %d, advancing",
                    state.supervision_round,
                )
                self._record_supervision_exit(
                    state,
                    method="delegation_no_directives",
                    overall_coverage="partial",
                    think_output=think_output,
                )
                break

            # Execute directives and merge results
            directive_results, round_new_sources, inline_stats = await self._execute_and_merge_directives(
                state, directives, timeout
            )
            total_new_sources += round_new_sources
            total_directives_executed += len(directive_results)

            # Post-round bookkeeping: think-after-results, history, save
            should_stop = await self._post_round_bookkeeping(
                state,
                directives,
                directive_results,
                think_output,
                round_new_sources,
                inline_stats,
                min_sources,
                timeout,
            )
            if should_stop:
                break

        return self._build_delegation_result(
            state,
            total_directives_executed,
            total_new_sources,
            phase_start_time,
        )

    def _prepare_round_coverage(
        self,
        state: DeepResearchState,
        min_sources: int,
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Prepare per-round coverage data: truncate messages, build coverage, compute delta."""
        if state.supervision_messages:
            state.supervision_messages = truncate_supervision_messages(
                state.supervision_messages,
                model=state.supervision_model,
            )

        coverage_data = self._build_per_query_coverage(state)
        coverage_delta: Optional[str] = None
        if state.supervision_round > 0:
            coverage_delta = self._compute_coverage_delta(
                state,
                coverage_data,
                min_sources=min_sources,
            )
        self._store_coverage_snapshot(state, coverage_data, suffix="pre")
        return coverage_data, coverage_delta

    @staticmethod
    def _advance_supervision_round(state: DeepResearchState) -> None:
        """Advance the supervision round counter.

        Centralises the round lifecycle so every code-path that completes
        a supervision round goes through one method.
        """
        state.supervision_round += 1

    def _record_supervision_exit(
        self,
        state: DeepResearchState,
        *,
        method: str,
        overall_coverage: str,
        think_output: Optional[str] = None,
        directives_generated: int = 0,
        directives_executed: int = 0,
        extra_data: Optional[dict[str, Any]] = None,
        audit_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a supervision early-exit: append history, trim, advance round."""
        if audit_data:
            self._write_audit_event(state, "supervision_result", data=audit_data)

        entry: dict[str, Any] = {
            "round": state.supervision_round,
            "method": method,
            "should_continue_gathering": False,
            "directives_generated": directives_generated,
            "overall_coverage": overall_coverage,
        }
        if directives_executed:
            entry["directives_executed"] = directives_executed
        if think_output is not None:
            entry["think_output"] = think_output[:_MAX_THINK_OUTPUT_STORED_CHARS]
        if extra_data:
            entry.update(extra_data)

        history = state.metadata.setdefault("supervision_history", [])
        history.append(entry)
        _trim_supervision_history(state)
        self._advance_supervision_round(state)

    def _build_delegation_result(
        self,
        state: DeepResearchState,
        total_directives_executed: int,
        total_new_sources: int,
        phase_start_time: float,
    ) -> WorkflowResult:
        """Finalize the delegation loop: save state, audit, and return result.

        Args:
            state: Current research state.
            total_directives_executed: Cumulative directives executed across rounds.
            total_new_sources: Cumulative new sources across rounds.
            phase_start_time: ``time.perf_counter()`` snapshot from phase start.

        Returns:
            WorkflowResult with supervision round metadata.
        """
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
    # Delegation loop sub-methods
    # ------------------------------------------------------------------

    def _should_exit_wall_clock(
        self,
        state: DeepResearchState,
        wall_clock_start: float,
        wall_clock_limit: float,
    ) -> bool:
        """Check if the supervision phase has exceeded its wall-clock timeout."""
        elapsed = time.monotonic() - wall_clock_start
        if elapsed < wall_clock_limit:
            return False
        logger.warning(
            "Supervision phase wall-clock timeout: %.0fs elapsed >= %.0fs limit. Exiting after %d rounds.",
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
        return True

    def _should_exit_heuristic(
        self,
        state: DeepResearchState,
        min_sources: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if the heuristic indicates coverage is sufficient (round > 0 only).

        Pure predicate — returns the decision and supporting data without
        mutating *state*.  The caller is responsible for writing audit events,
        appending to supervision history, and advancing the round counter.

        Returns:
            ``(should_exit, heuristic_data)`` where *heuristic_data* is the
            dict produced by :func:`assess_coverage_heuristic` (empty dict
            when round == 0).
        """
        if state.supervision_round == 0:
            return False, {}
        heuristic = self._assess_coverage_heuristic(state, min_sources)
        if heuristic["should_continue_gathering"]:
            return False, heuristic
        return True, heuristic

    async def _run_think_delegate_step(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        coverage_delta: Optional[str],
        provider_id: Optional[str],
        timeout: float,
    ) -> tuple[Optional[str], list[ResearchDirective], bool]:
        """Run the think + delegate steps and inject messages into history.

        Handles both single-call and two-call modes.

        Returns:
            Tuple of (think_output, directives, research_complete)
        """
        use_single_call = getattr(
            self.config,
            "deep_research_supervision_single_call",
            False,
        )
        is_first_round = self._is_first_round_decomposition(state)

        if use_single_call and not is_first_round:
            # Single-call path: think+delegate merged into one LLM call
            (
                think_output,
                directives,
                research_complete,
                delegation_content,
            ) = await self._supervision_combined_think_delegate_step(
                state,
                coverage_data,
                provider_id,
                timeout,
            )
            if think_output:
                state.add_supervision_message(
                    {
                        "role": "assistant",
                        "type": "think",
                        "round": state.supervision_round,
                        "content": think_output,
                    }
                )
            if delegation_content:
                state.add_supervision_message(
                    {
                        "role": "assistant",
                        "type": "delegation",
                        "round": state.supervision_round,
                        "content": delegation_content,
                    }
                )
        else:
            # Two-call path: separate think then delegate
            think_output = await self._supervision_think_step(
                state,
                coverage_data,
                timeout,
                coverage_delta=coverage_delta,
            )
            if think_output:
                state.add_supervision_message(
                    {
                        "role": "assistant",
                        "type": "think",
                        "round": state.supervision_round,
                        "content": think_output,
                    }
                )

            self._check_cancellation(state)
            directives, research_complete, delegation_content = await self._supervision_delegate_step(
                state,
                coverage_data,
                think_output,
                provider_id,
                timeout,
            )
            if delegation_content:
                state.add_supervision_message(
                    {
                        "role": "assistant",
                        "type": "delegation",
                        "round": state.supervision_round,
                        "content": delegation_content,
                    }
                )

        return think_output, directives, research_complete

    async def _execute_and_merge_directives(
        self,
        state: DeepResearchState,
        directives: list[ResearchDirective],
        timeout: float,
    ) -> tuple[list[TopicResearchResult], int, dict[str, Any]]:
        """Execute directives as parallel topic researchers and merge results.

        Handles directive execution, inline compression, findings accumulation,
        raw notes aggregation/trimming, and evidence inventory appending.

        Returns:
            Tuple of (directive_results, round_new_sources, inline_stats)
        """
        # Store directives for audit (capped to limit state serialization growth)
        state.directives.extend(directives)
        state.directives = state.directives[-_MAX_STORED_DIRECTIVES:]

        self._check_cancellation(state)
        directive_results = await self._execute_directives_async(
            state,
            directives,
            timeout,
        )

        round_new_sources = sum(r.sources_found for r in directive_results)

        # Inline compression of directive results
        inline_stats = await self._compress_directive_results_inline(
            state,
            directive_results,
            timeout,
        )

        # Accumulate findings as tool-result messages
        for result in directive_results:
            content = result.compressed_findings
            if not content and result.source_ids:
                content = self._build_directive_fallback_summary(
                    result,
                    state,
                )
            if content:
                state.add_supervision_message(
                    {
                        "role": "tool_result",
                        "type": "research_findings",
                        "round": state.supervision_round,
                        "directive_id": result.sub_query_id,
                        "content": content,
                    }
                )

        # Aggregate raw notes
        for result in directive_results:
            if result.raw_notes:
                state.raw_notes.append(result.raw_notes)

        # Trim raw_notes if they exceed the cap
        self._trim_raw_notes(state)

        # Append evidence inventories
        for result in directive_results:
            if result.raw_notes or result.source_ids:
                inventory = self._build_evidence_inventory(result, state)
                if inventory:
                    state.add_supervision_message(
                        {
                            "role": "tool_result",
                            "type": "evidence_inventory",
                            "round": state.supervision_round,
                            "directive_id": result.sub_query_id,
                            "content": inventory,
                        }
                    )

        return directive_results, round_new_sources, inline_stats

    def _trim_raw_notes(self, state: DeepResearchState) -> None:
        """Trim raw_notes if they exceed count or character caps."""
        notes_trimmed = 0
        # Trim by count — drop oldest entries via slice instead of O(n²) pop(0)
        if len(state.raw_notes) > _MAX_RAW_NOTES:
            excess = len(state.raw_notes) - _MAX_RAW_NOTES
            state.raw_notes = state.raw_notes[excess:]
            notes_trimmed += excess
        # Trim by total character count — compute drop count, then slice
        total_chars = sum(len(n) for n in state.raw_notes)
        if total_chars > _MAX_RAW_NOTES_CHARS:
            drop_count = 0
            while drop_count < len(state.raw_notes) and total_chars > _MAX_RAW_NOTES_CHARS:
                total_chars -= len(state.raw_notes[drop_count])
                drop_count += 1
            state.raw_notes = state.raw_notes[drop_count:]
            notes_trimmed += drop_count
        if notes_trimmed > 0:
            logger.warning(
                "Trimmed %d oldest raw_notes entries (count cap=%d, char cap=%d). %d entries remain (%d chars).",
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

    async def _post_round_bookkeeping(
        self,
        state: DeepResearchState,
        directives: list[ResearchDirective],
        directive_results: list[TopicResearchResult],
        think_output: Optional[str],
        round_new_sources: int,
        inline_stats: dict[str, Any],
        min_sources: int,
        timeout: float,
    ) -> bool:
        """Post-execution think, history recording, state saving.

        Returns:
            True if the loop should stop (no new sources this round)
        """
        # Think-after-results: assess what was learned
        post_think_output: Optional[str] = None
        if directive_results:
            post_coverage_data = self._build_per_query_coverage(state)
            post_delta = self._compute_coverage_delta(
                state,
                post_coverage_data,
                min_sources=min_sources,
            )
            self._store_coverage_snapshot(state, post_coverage_data, suffix="post")
            post_think_output = await self._supervision_think_step(
                state,
                post_coverage_data,
                timeout,
                coverage_delta=post_delta,
            )
            if post_think_output:
                state.add_supervision_message(
                    {
                        "role": "assistant",
                        "type": "think",
                        "round": state.supervision_round,
                        "content": post_think_output,
                    }
                )

        # Record history and advance round
        history = state.metadata.setdefault("supervision_history", [])
        history.append(
            {
                "round": state.supervision_round,
                "method": "delegation",
                "should_continue_gathering": round_new_sources > 0,
                "directives_generated": len(directives),
                "directives_executed": len(directive_results),
                "new_sources": round_new_sources,
                "think_output": (think_output or "")[:_MAX_THINK_OUTPUT_STORED_CHARS],
                "post_execution_think": (post_think_output or "")[:_MAX_THINK_OUTPUT_STORED_CHARS],
                "directive_topics": [d.research_topic[:100] for d in directives],
                "inline_compression": inline_stats,
            }
        )
        _trim_supervision_history(state)

        # Truncate supervision messages *before* persisting state so that
        # in-memory growth from directive results + post-think is bounded
        # within the same round (not deferred to the next round's guard).
        if state.supervision_messages:
            state.supervision_messages = truncate_supervision_messages(
                state.supervision_messages,
                model=state.supervision_model,
            )

        self._advance_supervision_round(state)
        self.memory.save_deep_research(state)

        # If no new sources were found this round, stop delegating
        if round_new_sources == 0:
            logger.info(
                "Supervision delegation: no new sources in round %d, stopping",
                state.supervision_round,
            )
            return True
        return False

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
                state,
                coverage_data,
                coverage_delta=coverage_delta,
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
            timeout=self.config.deep_research_reflection_timeout,
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
                state,
                think_output,
                provider_id,
                timeout,
            )

        system_prompt = self._build_delegation_system_prompt()
        user_prompt = self._build_delegation_user_prompt(
            state,
            coverage_data,
            think_output,
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
                call_result.result.content,
                state,
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
            state,
            coverage_data,
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
                raw_content,
                state,
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
        return build_combined_think_delegate_system_prompt()

    def _build_combined_think_delegate_user_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
    ) -> str:
        return build_combined_think_delegate_user_prompt(state, coverage_data)

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
            self.config,
            "deep_research_max_concurrent_research_units",
            5,
        )
        topic_max_searches = getattr(
            self.config,
            "deep_research_topic_max_tool_calls",
            10,
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
                result = await self._execute_topic_research_async(
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
                # Incremental persistence after each researcher completes
                # so accumulated sources survive crashes or premature stale detection
                try:
                    async with state_lock:
                        self.memory.save_deep_research(state)
                except Exception:
                    logger.debug("Incremental persist failed for %s", sq.id)
                return result
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
            state.add_topic_research_result(result)

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
        results_to_compress = [r for r in directive_results if r.source_ids and r.compressed_findings is None]

        if not results_to_compress:
            already_compressed = sum(1 for r in directive_results if r.compressed_findings is not None)
            return {
                "compressed": 0,
                "failed": 0,
                "skipped": already_compressed,
            }

        # Per-result compression timeout — same as batch compression timeout
        compression_timeout: float = getattr(
            self.config,
            "deep_research_compression_timeout",
            120.0,
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
                if success:
                    # compressed_findings now captures essential content —
                    # free message_history to bound state memory growth.
                    topic_result.message_history.clear()
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
            summary = sanitize_external_content(topic_result.per_topic_summary)
            if len(summary) > max_chars:
                return summary[:max_chars] + "..."
            return summary

        # Build from source content
        topic_sources = [s for s in state.sources if s.id in topic_result.source_ids]
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
        topic_sources = [source_map[sid] for sid in topic_result.source_ids if sid in source_map]

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
            entry = f'- [{idx}] "{title}"'
            if domain:
                entry += f" ({domain})"
            if len(entry) + 1 > remaining:
                break
            parts.append(entry)
            remaining -= len(entry) + 1

        # Key findings from supervisor summary (structured for gap analysis)
        if topic_result.supervisor_summary:
            # Truncate to fit within remaining budget
            brief = sanitize_external_content(topic_result.supervisor_summary)
            label = "Key findings: "
            max_brief = remaining - len(label) - 1
            if max_brief > 20:
                if len(brief) > max_brief:
                    brief = brief[: max_brief - 3] + "..."
                findings_line = f"{label}{brief}"
                parts.append(findings_line)
                remaining -= len(findings_line) + 1

        # Data point estimate from raw notes (count paragraphs as proxy)
        if topic_result.raw_notes and remaining > 30:
            # Count non-empty lines as a rough data-point proxy
            lines = [ln for ln in topic_result.raw_notes.split("\n") if ln.strip()]
            data_points = min(len(lines), 999)
            dp_line = f"Key data points: ~{data_points} extracted"
            if len(dp_line) + 1 <= remaining:
                parts.append(dp_line)
                remaining -= len(dp_line) + 1

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[: max_chars - 3] + "..."
        return result

    # ------------------------------------------------------------------
    # Query complexity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_query_complexity(state: DeepResearchState) -> str:
        return classify_query_complexity(state)

    # ------------------------------------------------------------------
    # Delegation prompts
    # ------------------------------------------------------------------

    def _build_delegation_system_prompt(self) -> str:
        return build_delegation_system_prompt()

    def _build_delegation_user_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        think_output: Optional[str] = None,
    ) -> str:
        return build_delegation_user_prompt(state, coverage_data, think_output)

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
            self.config,
            "deep_research_max_concurrent_research_units",
            5,
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
            self.config,
            "deep_research_max_concurrent_research_units",
            5,
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

            directives.append(
                ResearchDirective(
                    research_topic=topic,
                    perspective=d.get("perspective", ""),
                    evidence_needed=d.get("evidence_needed", ""),
                    priority=priority,
                    supervision_round=state.supervision_round,
                )
            )

        return directives, False

    # ==================================================================
    # First-round decomposition prompts (supervisor-owned decomposition)
    # ==================================================================

    def _build_first_round_think_system_prompt(self) -> str:
        return build_first_round_think_system_prompt()

    def _build_first_round_think_prompt(self, state: DeepResearchState) -> str:
        return build_first_round_think_prompt(state)

    def _build_first_round_delegation_system_prompt(self) -> str:
        return build_first_round_delegation_system_prompt()

    def _build_first_round_delegation_user_prompt(
        self,
        state: DeepResearchState,
        think_output: Optional[str] = None,
    ) -> str:
        return build_first_round_delegation_user_prompt(state, think_output)

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
        initial_directives, research_complete, gen_content, should_skip = await self._run_first_round_generate(
            state,
            think_output,
            effective_provider,
            timeout,
        )
        if should_skip:
            return initial_directives, research_complete, gen_content

        # --- Call 2: Critique the initial directives ---
        initial_count = len(initial_directives)
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

        critique_text, needs_revision, should_return_initial = await self._run_first_round_critique(
            state,
            initial_count,
            directives_json,
            effective_provider,
            timeout,
        )
        if should_return_initial:
            return initial_directives, research_complete, gen_content

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
            return initial_directives, research_complete, gen_content

        return await self._run_first_round_revise(
            state,
            initial_directives,
            initial_count,
            directives_json,
            critique_text,
            research_complete,
            effective_provider,
            timeout,
            gen_content=gen_content,
        )

    async def _run_first_round_generate(
        self,
        state: DeepResearchState,
        think_output: Optional[str],
        effective_provider: Optional[str],
        timeout: float,
    ) -> tuple[list[ResearchDirective], bool, Optional[str], bool]:
        """Run the generation LLM call for first-round decomposition.

        Args:
            state: Current research state.
            think_output: Decomposition strategy from think step.
            effective_provider: Resolved LLM provider ID.
            timeout: Request timeout.

        Returns:
            Tuple of ``(initial_directives, research_complete, raw_content,
            should_skip)`` where *should_skip* is ``True`` when the caller
            should return early (research_complete or no directives).
        """
        self._check_cancellation(state)
        gen_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate_generate",
            system_prompt=self._build_first_round_delegation_system_prompt(),
            user_prompt=self._build_first_round_delegation_user_prompt(
                state,
                think_output,
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
                "First-round generate call failed: %s",
                gen_result.error,
            )
            return [], False, None, True

        # Extract initial directives
        if gen_result.parsed is not None:
            gen_delegation: DelegationResponse = gen_result.parsed
            initial_directives = self._apply_directive_caps(
                gen_delegation.directives,
                state,
            )
            research_complete = gen_delegation.research_complete
        else:
            logger.warning(
                "First-round generate parse failed, falling back to legacy parser",
            )
            initial_directives, research_complete = self._parse_delegation_response(
                gen_result.result.content,
                state,
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
                "directive_topics": [d.research_topic[:100] for d in initial_directives],
            },
        )

        # If generation signalled research_complete or produced no directives,
        # skip critique/revision — there's nothing to refine.
        should_skip = research_complete or not initial_directives
        if should_skip:
            self._write_audit_event(
                state,
                "first_round_decomposition",
                data={
                    "initial_directive_count": initial_count,
                    "final_directive_count": initial_count,
                    "critique_triggered_revision": False,
                    "skip_reason": ("research_complete" if research_complete else "no_directives"),
                },
            )

        return initial_directives, research_complete, gen_result.result.content, should_skip

    async def _run_first_round_critique(
        self,
        state: DeepResearchState,
        initial_count: int,
        directives_json: str,
        effective_provider: Optional[str],
        timeout: float,
    ) -> tuple[str, bool, bool]:
        """Run the critique LLM call for first-round decomposition.

        Args:
            state: Current research state.
            initial_count: Number of directives from the generate step.
            directives_json: JSON-serialized directives for the critique prompt.
            effective_provider: Resolved LLM provider ID.
            timeout: Request timeout.

        Returns:
            Tuple of ``(critique_text, needs_revision, should_return_initial)``
            where *should_return_initial* is ``True`` when the critique call
            failed and the caller should fall back to the initial directives.
        """
        self._check_cancellation(state)

        critique_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate_critique",
            system_prompt=self._build_critique_system_prompt(),
            user_prompt=self._build_critique_user_prompt(
                state,
                directives_json,
            ),
            provider_id=effective_provider,
            model=state.supervision_model,
            temperature=0.2,
            timeout=getattr(
                self.config,
                "deep_research_reflection_timeout",
                60.0,
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
            return "", False, True

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

        return critique_text, needs_revision, False

    async def _run_first_round_revise(
        self,
        state: DeepResearchState,
        initial_directives: list[ResearchDirective],
        initial_count: int,
        directives_json: str,
        critique_text: str,
        research_complete: bool,
        effective_provider: Optional[str],
        timeout: float,
        *,
        gen_content: Optional[str] = None,
    ) -> tuple[list[ResearchDirective], bool, Optional[str]]:
        """Run the revision LLM call for first-round decomposition.

        Args:
            state: Current research state.
            initial_directives: Directives from the generate step (fallback).
            initial_count: Number of initial directives.
            directives_json: JSON-serialized initial directives.
            critique_text: Critique output to inform revision.
            research_complete: Research-complete flag from generate step.
            effective_provider: Resolved LLM provider ID.
            timeout: Request timeout.
            gen_content: Raw content from the generate step (used as fallback
                when the revision call fails).

        Returns:
            Tuple of ``(final_directives, research_complete, raw_content)``.
        """
        self._check_cancellation(state)
        revise_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="supervision_delegate_revise",
            system_prompt=self._build_revision_system_prompt(),
            user_prompt=self._build_revision_user_prompt(
                state,
                directives_json,
                critique_text,
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
            return initial_directives, research_complete, gen_content

        # Extract revised directives
        if revise_result.parsed is not None:
            rev_delegation: DelegationResponse = revise_result.parsed
            final_directives = self._apply_directive_caps(
                rev_delegation.directives,
                state,
            )
            research_complete = rev_delegation.research_complete
        else:
            logger.warning(
                "Revision parse failed, falling back to legacy parser",
            )
            final_directives, research_complete = self._parse_delegation_response(
                revise_result.result.content,
                state,
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
                "directive_topics": [d.research_topic[:100] for d in final_directives],
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
        return build_critique_system_prompt()

    def _build_critique_user_prompt(
        self,
        state: DeepResearchState,
        directives_json: str,
    ) -> str:
        return build_critique_user_prompt(state, directives_json)

    def _build_revision_system_prompt(self) -> str:
        return build_revision_system_prompt()

    def _build_revision_user_prompt(
        self,
        state: DeepResearchState,
        directives_json: str,
        critique_text: str,
    ) -> str:
        return build_revision_user_prompt(state, directives_json, critique_text)

    @staticmethod
    def _critique_has_issues(critique_text: str) -> bool:
        return critique_has_issues(critique_text)

    # ==================================================================
    # Shared helpers (used by delegation model)
    # ==================================================================

    def _build_per_query_coverage(
        self,
        state: DeepResearchState,
    ) -> list[dict[str, Any]]:
        return build_per_query_coverage(state)

    def _store_coverage_snapshot(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        suffix: Optional[str] = None,
    ) -> None:
        store_coverage_snapshot(state, coverage_data, suffix=suffix)

    def _compute_coverage_delta(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        min_sources: int = 3,
    ) -> Optional[str]:
        return compute_coverage_delta(state, coverage_data, min_sources)

    def _build_think_prompt(
        self,
        state: DeepResearchState,
        coverage_data: list[dict[str, Any]],
        coverage_delta: Optional[str] = None,
    ) -> str:
        return build_think_prompt(state, coverage_data, coverage_delta)

    def _build_think_system_prompt(self) -> str:
        return build_think_system_prompt()

    def _assess_coverage_heuristic(
        self,
        state: DeepResearchState,
        min_sources: int,
    ) -> dict[str, Any]:
        return assess_coverage_heuristic(state, min_sources, self.config)
