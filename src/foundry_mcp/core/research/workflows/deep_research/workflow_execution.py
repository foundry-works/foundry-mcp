"""Async workflow execution engine for deep research.

Orchestrates the multi-phase workflow (planning, gathering, analysis,
synthesis, refinement) with cancellation support, error handling,
and resource cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_hostname,
)

logger = logging.getLogger(__name__)


class WorkflowExecutionMixin:
    """Mixin providing async workflow execution for deep research.

    Requires the composing class to provide:
    - self.config: ResearchConfig
    - self.memory: ResearchMemory
    - self.hooks: SupervisorHooks
    - self.orchestrator: SupervisorOrchestrator
    - self._tasks: dict[str, BackgroundTask]
    - self._tasks_lock: threading.Lock
    - self._search_providers: dict[str, SearchProvider]
    - self._write_audit_event(): from AuditMixin
    - self._flush_state(): from PersistenceMixin
    - self._record_workflow_error(): from ErrorHandlingMixin
    - self._safe_orchestrator_transition(): from ErrorHandlingMixin
    - self._check_cancellation(): defined here
    - Phase execution methods from phase mixins
    """

    config: Any
    memory: Any
    hooks: Any
    orchestrator: Any
    _tasks: dict[str, Any]
    _tasks_lock: threading.Lock
    _search_providers: dict[str, Any]

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _flush_state(self, *args: Any, **kwargs: Any) -> None: ...
        def _record_workflow_error(self, *args: Any, **kwargs: Any) -> None: ...
        def _safe_orchestrator_transition(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_clarification_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_planning_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_gathering_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_analysis_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_synthesis_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_refinement_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_extract_followup_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_digest_step_async(self, *args: Any, **kwargs: Any) -> Any: ...

    def _check_cancellation(self, state: DeepResearchState) -> None:
        """Check if cancellation has been requested for this research session.

        Raises:
            asyncio.CancelledError: If cancellation is detected
        """
        # Retrieve the background task for this research session
        with self._tasks_lock:
            bg_task = self._tasks.get(state.id)

        if bg_task and bg_task.is_cancelled:
            logger.info(
                "Cancellation detected for research %s at phase %s, iteration %d",
                state.id,
                state.phase.value,
                state.iteration,
            )
            raise asyncio.CancelledError("Cancellation requested")

    async def _run_phase(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
        executor: Any,
        *,
        skip_error_check: bool = False,
        skip_transition: bool = False,
    ) -> WorkflowResult | None:
        """Execute common phase lifecycle: cancel -> timer -> hooks -> audit -> execute -> error -> hooks -> audit -> transition.

        Encapsulates the boilerplate shared across all 5 phase dispatch blocks
        in ``_execute_workflow_async``.

        Args:
            state: Current research state.
            phase: The phase being executed (used for audit events and orchestrator transition).
            executor: An *unawaited* coroutine returned by ``_execute_<phase>_async(...)``.
            skip_error_check: If True, do not check ``result.success`` for failure
                (used by REFINEMENT which always succeeds).
            skip_transition: If True, skip the standard orchestrator transition
                (used by SYNTHESIS/REFINEMENT which have custom post-processing).

        Returns:
            ``WorkflowResult`` on phase failure (caller should ``return`` it),
            ``None`` on success (caller continues to next phase).
        """
        self._check_cancellation(state)
        phase_started = time.perf_counter()
        self.hooks.emit_phase_start(state)
        self._write_audit_event(
            state,
            "phase_start",
            data={"phase": state.phase.value},
        )

        result = await executor

        if not skip_error_check and not result.success:
            self._write_audit_event(
                state,
                "phase_error",
                data={"phase": state.phase.value, "error": result.error},
                level="error",
            )
            state.mark_failed(result.error or f"Phase {state.phase.value} failed")
            self._flush_state(state)
            return result

        self.hooks.emit_phase_complete(state)
        self._write_audit_event(
            state,
            "phase_complete",
            data={
                "phase": state.phase.value,
                "duration_ms": (time.perf_counter() - phase_started) * 1000,
            },
        )

        if not skip_transition:
            self._safe_orchestrator_transition(state, phase)

        return None

    async def _execute_workflow_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
    ) -> WorkflowResult:
        """Execute the full workflow asynchronously.

        This is the main async entry point that orchestrates all phases.
        """
        start_time = time.perf_counter()

        try:
            # Phase execution based on current state
            if state.phase == DeepResearchPhase.CLARIFICATION:
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.CLARIFICATION,
                    self._execute_clarification_async(
                        state=state,
                        provider_id=(
                            getattr(self.config, "deep_research_clarification_provider", None)
                            or self.config.default_provider
                        ),
                        timeout=self.config.get_phase_timeout("planning"),  # Reuse planning timeout
                    ),
                )
                if err:
                    return err

            if state.phase == DeepResearchPhase.PLANNING:
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.PLANNING,
                    self._execute_planning_async(
                        state=state,
                        provider_id=state.planning_provider,
                        timeout=self.config.get_phase_timeout("planning"),
                    ),
                )
                if err:
                    return err

            if state.phase == DeepResearchPhase.GATHERING:
                # Mark the current iteration as in progress (for cancellation handling)
                state.metadata["iteration_in_progress"] = True
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.GATHERING,
                    self._execute_gathering_async(
                        state=state,
                        provider_id=provider_id,
                        timeout=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    ),
                )
                if err:
                    return err

                # Optional: Execute extract follow-up to expand URL content
                if self.config.tavily_extract_in_deep_research:
                    extract_result = await self._execute_extract_followup_async(
                        state=state,
                        max_urls=self.config.tavily_extract_max_urls,
                    )
                    if extract_result:
                        self._write_audit_event(
                            state,
                            "extract_followup_complete",
                            data={
                                "urls_extracted": extract_result.get("urls_extracted", 0),
                                "urls_failed": extract_result.get("urls_failed", 0),
                            },
                        )

                # Proactive digest: digest sources immediately after gathering
                # when policy is "proactive", ensuring uniform pre-processed
                # content before the analysis phase.
                if self.config.deep_research_digest_policy == "proactive":
                    self._check_cancellation(state)
                    logger.info(
                        "Running proactive digest on %d sources for research %s",
                        len(state.sources),
                        state.id,
                    )
                    digest_stats = await self._execute_digest_step_async(
                        state=state,
                        query=state.original_query,
                    )
                    self._write_audit_event(
                        state,
                        "proactive_digest_complete",
                        data={
                            "sources_digested": digest_stats.get("sources_digested", 0),
                            "sources_selected": digest_stats.get("sources_selected", 0),
                            "sources_ranked": digest_stats.get("sources_ranked", 0),
                            "errors": len(digest_stats.get("digest_errors", [])),
                        },
                    )
                    # Persist state with digested content
                    self.memory.save_deep_research(state)

            if state.phase == DeepResearchPhase.ANALYSIS:
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.ANALYSIS,
                    self._execute_analysis_async(
                        state=state,
                        provider_id=state.analysis_provider,
                        timeout=self.config.get_phase_timeout("analysis"),
                    ),
                )
                if err:
                    return err

            if state.phase == DeepResearchPhase.SYNTHESIS:
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.SYNTHESIS,
                    self._execute_synthesis_async(
                        state=state,
                        provider_id=state.synthesis_provider,
                        timeout=self.config.get_phase_timeout("synthesis"),
                    ),
                    skip_transition=True,
                )
                if err:
                    return err

                # Phase-specific: custom orchestrator + iteration decision
                try:
                    self.orchestrator.evaluate_phase_completion(state, DeepResearchPhase.SYNTHESIS)
                    self.orchestrator.decide_iteration(state)
                    prompt = self.orchestrator.get_reflection_prompt(state, DeepResearchPhase.SYNTHESIS)
                    self.hooks.think_pause(state, prompt)
                    self.orchestrator.record_to_state(state)
                except Exception as exc:
                    logger.exception(
                        "Orchestrator transition failed for synthesis, research %s: %s",
                        state.id,
                        exc,
                    )
                    self._write_audit_event(
                        state,
                        "orchestrator_error",
                        data={
                            "phase": "synthesis",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        level="error",
                    )
                    self._record_workflow_error(exc, state, "orchestrator_synthesis")
                    raise

                # Check if refinement needed
                if state.should_continue_refinement():
                    state.phase = DeepResearchPhase.REFINEMENT
                else:
                    # Mark iteration as successfully completed (no more refinement)
                    state.metadata["iteration_in_progress"] = False
                    state.metadata["last_completed_iteration"] = state.iteration
                    state.mark_completed(report=state.report)

            # Handle refinement phase
            if state.phase == DeepResearchPhase.REFINEMENT:
                # Mark the current iteration as in progress (for cancellation handling)
                state.metadata["iteration_in_progress"] = True
                await self._run_phase(
                    state,
                    DeepResearchPhase.REFINEMENT,
                    self._execute_refinement_async(
                        state=state,
                        provider_id=state.refinement_provider,
                        timeout=self.config.get_phase_timeout("refinement"),
                    ),
                    skip_error_check=True,
                    skip_transition=True,
                )

                # Mark iteration as successfully completed
                state.metadata["iteration_in_progress"] = False
                state.metadata["last_completed_iteration"] = state.iteration

                if state.should_continue_refinement():
                    # Check for cancellation before starting new iteration
                    self._check_cancellation(state)
                    state.start_new_iteration()
                    # Recursively continue workflow
                    return await self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                else:
                    state.mark_completed(report=state.report)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            state.total_duration_ms += duration_ms

            # Flush final state (bypasses throttle to ensure completion is captured)
            self._flush_state(state)
            self._write_audit_event(
                state,
                "workflow_complete",
                data={
                    "success": True,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "sub_query_count": len(state.sub_queries),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "report_length": len(state.report or ""),
                    # Existing totals
                    "total_tokens_used": state.total_tokens_used,
                    "total_duration_ms": state.total_duration_ms,
                    # Token breakdown totals
                    "total_input_tokens": sum(m.input_tokens for m in state.phase_metrics),
                    "total_output_tokens": sum(m.output_tokens for m in state.phase_metrics),
                    "total_cached_tokens": sum(m.cached_tokens for m in state.phase_metrics),
                    # Per-phase metrics
                    "phase_metrics": [
                        {
                            "phase": m.phase,
                            "duration_ms": m.duration_ms,
                            "input_tokens": m.input_tokens,
                            "output_tokens": m.output_tokens,
                            "cached_tokens": m.cached_tokens,
                            "provider_id": m.provider_id,
                            "model_used": m.model_used,
                        }
                        for m in state.phase_metrics
                    ],
                    # Search provider stats
                    "search_provider_stats": state.search_provider_stats,
                    "total_search_queries": sum(state.search_provider_stats.values()),
                    # Source hostnames
                    "source_hostnames": sorted(
                        set(h for s in state.sources if s.url and (h := _extract_hostname(s.url)))
                    ),
                    # Research mode
                    "research_mode": state.research_mode.value,
                },
            )

            return WorkflowResult(
                success=True,
                content=state.report or "Research completed",
                provider_id=provider_id,
                tokens_used=state.total_tokens_used,
                duration_ms=duration_ms,
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "sub_query_count": len(state.sub_queries),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "is_complete": state.completed_at is not None,
                },
            )

        except asyncio.CancelledError:
            # Handle cancellation: implement partial result policy
            # Discard incomplete iteration results, persist only completed iterations

            # Transition to "cancelling" state
            state.metadata["cancellation_state"] = "cancelling"
            logger.info(
                "Workflow entering cancelling state for research %s",
                state.id,
            )

            logger.warning(
                "Workflow cancelled at phase %s, iteration %d, research %s",
                state.phase.value,
                state.iteration,
                state.id,
            )
            state.metadata["cancelled"] = True

            # Check if current iteration is incomplete
            if state.metadata.get("iteration_in_progress"):
                # Current iteration is incomplete - discard partial results from this iteration
                last_completed_iteration = state.metadata.get("last_completed_iteration")
                if last_completed_iteration is not None and last_completed_iteration < state.iteration:
                    # We have a safe checkpoint from a prior completed iteration
                    logger.info(
                        "Discarding partial results from incomplete iteration %d (last completed: %d), research %s",
                        state.iteration,
                        last_completed_iteration,
                        state.id,
                    )
                    # Rollback state to last completed iteration by restoring from checkpoint
                    # For now, mark that we need to discard this iteration on resume
                    state.metadata["discarded_iteration"] = state.iteration
                    state.iteration = last_completed_iteration
                    state.phase = DeepResearchPhase.SYNTHESIS
                else:
                    # First iteration is incomplete - we cannot safely resume, must discard entire session
                    logger.warning(
                        "First iteration incomplete at cancellation, marking session for discard, research %s",
                        state.id,
                    )
                    state.metadata["discarded_iteration"] = state.iteration
            else:
                # Iteration was successfully completed, safe to save
                logger.info(
                    "Cancelled after completed iteration %d, research %s",
                    state.iteration,
                    state.id,
                )

            # Save state with cancelling transition
            self.memory.save_deep_research(state)

            # Transition to "cleanup" state before cleanup phase
            state.metadata["cancellation_state"] = "cleanup"
            logger.info(
                "Workflow entering cleanup state for research %s",
                state.id,
            )

            # Mark the state as cancelled with phase context
            state.mark_cancelled(phase_state=f"phase={state.phase.value}, iteration={state.iteration}")
            self.memory.save_deep_research(state)

            self._write_audit_event(
                state,
                "workflow_cancelled",
                data={
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "iteration_in_progress": state.metadata.get("iteration_in_progress"),
                    "last_completed_iteration": state.metadata.get("last_completed_iteration"),
                    "discarded_iteration": state.metadata.get("discarded_iteration"),
                    "cancellation_state": state.metadata.get("cancellation_state"),
                    "terminal_status": "cancelled",
                },
                level="warning",
            )
            # Re-raise to propagate cancellation to caller
            raise
        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.exception(
                "Workflow execution failed at phase %s, iteration %d: %s",
                state.phase.value,
                state.iteration,
                exc,
            )
            if not state.metadata.get("failed"):
                state.mark_failed(str(exc))
            self.memory.save_deep_research(state)
            self._write_audit_event(
                state,
                "workflow_error",
                data={
                    "error": str(exc),
                    "traceback": tb_str,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
                level="error",
            )
            self._record_workflow_error(exc, state, "workflow_execution")
            return WorkflowResult(
                success=False,
                content="",
                error=str(exc),
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
            )
        finally:
            # Ensure resources are cleaned up on cancellation, timeout, or any other exit
            # This block runs regardless of exception type or successful completion,
            # but does not re-save state if already saved (to avoid duplicate saves)
            logger.debug(
                "Workflow cleanup phase for research %s at phase %s",
                state.id,
                state.phase.value,
            )

            # Close any open search provider connections
            # (Currently search providers don't maintain persistent connections,
            # but this is in place for future stateful provider implementations)
            for provider in self._search_providers.values():
                try:
                    # Check if provider has async close method
                    if hasattr(provider, "aclose"):
                        await provider.aclose()
                    elif hasattr(provider, "close"):
                        provider.close()
                except Exception as cleanup_exc:
                    logger.warning(
                        "Error closing search provider during cleanup: %s",
                        cleanup_exc,
                    )

            # After cleanup completes, mark cancellation as fully complete if transitioning through cleanup state
            if state.metadata.get("cancellation_state") == "cleanup":
                state.metadata["cancellation_state"] = "cancelled"
                logger.info(
                    "Workflow cancellation complete for research %s",
                    state.id,
                )
                self.memory.save_deep_research(state)
