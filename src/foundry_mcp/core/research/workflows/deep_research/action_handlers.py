"""Action handlers for deep research workflow.

Implements the start, continue, status, report, and cancel actions
that form the public API surface of the deep research workflow.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core import task_registry
from foundry_mcp.core.research.models.deep_research import (
    DEFAULT_MAX_SUPERVISION_ROUNDS,
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.sources import ResearchMode
from foundry_mcp.core.research.workflows.base import MAX_PROMPT_LENGTH, WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._constants import (
    MAX_CONCURRENT_PROVIDERS,
    MAX_ITERATIONS,
    MAX_SOURCES_PER_QUERY,
    MAX_SUB_QUERIES,
)
from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
    _active_research_sessions,
    _active_sessions_lock,
)

logger = logging.getLogger(__name__)


class ActionHandlersMixin:
    """Mixin providing action handlers for deep research workflow.

    Requires the composing class to provide:
    - self.config: ResearchConfig
    - self.memory: ResearchMemory
    - self._write_audit_event(): from AuditMixin
    - self._persist_state_if_needed(): from PersistenceMixin
    - self._flush_state(): from PersistenceMixin
    - self.get_background_task(): from BackgroundTaskMixin
    - self._start_background_task(): from BackgroundTaskMixin
    - self._cleanup_completed_task(): from BackgroundTaskMixin
    - self._execute_workflow_async(): from WorkflowExecutionMixin
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _persist_state_if_needed(self, *args: Any, **kwargs: Any) -> None: ...
        def _flush_state(self, *args: Any, **kwargs: Any) -> None: ...
        def get_background_task(self, *args: Any, **kwargs: Any) -> Any: ...
        def _start_background_task(self, *args: Any, **kwargs: Any) -> Any: ...
        def _cleanup_completed_task(self, *args: Any, **kwargs: Any) -> None: ...
        async def _execute_workflow_async(self, *args: Any, **kwargs: Any) -> Any: ...

    def _start_research(
        self,
        query: Optional[str],
        provider_id: Optional[str],
        system_prompt: Optional[str],
        max_iterations: int,
        max_sub_queries: int,
        max_sources_per_query: int,
        follow_links: bool,
        timeout_per_operation: float,
        max_concurrent: int,
        background: bool,
        task_timeout: Optional[float],
    ) -> WorkflowResult:
        """Start a new deep research session."""
        if not query:
            return WorkflowResult(
                success=False,
                content="",
                error="Query is required to start research",
            )

        # Input bounds validation
        violations: list[str] = []
        if len(query) > MAX_PROMPT_LENGTH:
            violations.append(f"query length {len(query)} exceeds maximum {MAX_PROMPT_LENGTH} characters")
        if max_iterations > MAX_ITERATIONS:
            violations.append(f"max_iterations {max_iterations} exceeds maximum {MAX_ITERATIONS}")
        if max_sub_queries > MAX_SUB_QUERIES:
            violations.append(f"max_sub_queries {max_sub_queries} exceeds maximum {MAX_SUB_QUERIES}")
        if max_sources_per_query > MAX_SOURCES_PER_QUERY:
            violations.append(f"max_sources_per_query {max_sources_per_query} exceeds maximum {MAX_SOURCES_PER_QUERY}")
        if max_concurrent > MAX_CONCURRENT_PROVIDERS:
            violations.append(f"max_concurrent {max_concurrent} exceeds maximum {MAX_CONCURRENT_PROVIDERS}")
        if violations:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Input validation failed: {'; '.join(violations)}",
                metadata={"validation_errors": violations},
            )

        # Resolve per-phase providers and models from config
        # Supports ProviderSpec format: "[cli]gemini:pro" -> (provider_id, model)
        planning_pid, planning_model = self.config.resolve_phase_provider("planning")
        synthesis_pid, synthesis_model = self.config.resolve_phase_provider("synthesis")

        # Determine initial phase: CLARIFICATION if enabled, else SUPERVISION
        initial_phase = DeepResearchPhase.SUPERVISION
        if getattr(self.config, "deep_research_allow_clarification", False):
            initial_phase = DeepResearchPhase.CLARIFICATION

        # Create initial state with per-phase provider configuration
        state = DeepResearchState(
            original_query=query,
            phase=initial_phase,
            max_iterations=max_iterations,
            max_sub_queries=max_sub_queries,
            max_sources_per_query=max_sources_per_query,
            follow_links=follow_links,
            research_mode=ResearchMode(self.config.deep_research_mode),
            system_prompt=system_prompt,
            # Per-phase providers: explicit provider_id overrides config
            planning_provider=provider_id or planning_pid,
            synthesis_provider=provider_id or synthesis_pid,
            # Per-phase models from ProviderSpec (only used if provider_id not overridden)
            planning_model=None if provider_id else planning_model,
            synthesis_model=None if provider_id else synthesis_model,
            # Supervision configuration
            max_supervision_rounds=getattr(self.config, "deep_research_max_supervision_rounds", DEFAULT_MAX_SUPERVISION_ROUNDS),
        )

        # Save initial state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "workflow_start",
            data={
                "query": state.original_query,
                "config": {
                    "max_iterations": max_iterations,
                    "max_sub_queries": max_sub_queries,
                    "max_sources_per_query": max_sources_per_query,
                    "follow_links": follow_links,
                    "timeout_per_operation": timeout_per_operation,
                    "max_concurrent": max_concurrent,
                },
                "provider_id": provider_id,
                "background": background,
                "task_timeout": task_timeout,
            },
        )

        if background:
            return self._start_background_task(
                state=state,
                provider_id=provider_id,
                timeout_per_operation=timeout_per_operation,
                max_concurrent=max_concurrent,
                task_timeout=task_timeout,
            )

        # Synchronous execution with optional timeout enforcement
        return self._run_sync(
            state=state,
            provider_id=provider_id,
            timeout_per_operation=timeout_per_operation,
            max_concurrent=max_concurrent,
            task_timeout=task_timeout,
        )

    def _continue_research(
        self,
        research_id: Optional[str],
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
        background: bool = False,
        task_timeout: Optional[float] = None,
    ) -> WorkflowResult:
        """Continue an existing research session.

        Args:
            research_id: ID of the research session to continue
            provider_id: Optional provider ID for LLM calls
            timeout_per_operation: Timeout per operation in seconds
            max_concurrent: Maximum concurrent operations
            background: If True, run in background thread (default: False)
            task_timeout: Overall timeout for background task (optional)

        Returns:
            WorkflowResult with research state or error
        """
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required to continue research",
            )

        # Load existing state
        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        if state.completed_at is not None:
            return WorkflowResult(
                success=True,
                content=state.report or "Research already completed",
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "is_complete": True,
                },
            )

        # Run in background if requested
        if background:
            return self._start_background_task(
                state=state,
                provider_id=provider_id,
                timeout_per_operation=timeout_per_operation,
                max_concurrent=max_concurrent,
                task_timeout=task_timeout,
            )

        # Continue from current phase synchronously with optional timeout
        return self._run_sync(
            state=state,
            provider_id=provider_id,
            timeout_per_operation=timeout_per_operation,
            max_concurrent=max_concurrent,
            task_timeout=task_timeout,
        )

    def _get_status(self, research_id: Optional[str]) -> WorkflowResult:
        """Get the current status of a research session."""
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required",
            )

        # Check background task first
        bg_task = self.get_background_task(research_id)
        if bg_task:
            is_active = not bg_task.is_done
            # Prefer in-memory state for active tasks to avoid clobbering workflow saves.
            if is_active:
                with _active_sessions_lock:
                    state = _active_research_sessions.get(research_id)
            else:
                state = None
            if state is None:
                state = self.memory.load_deep_research(research_id)
            metadata: dict[str, Any] = {
                "research_id": research_id,
                "task_status": bg_task.status.value,
                "elapsed_ms": bg_task.elapsed_ms,
                "is_complete": bg_task.is_done,
            }
            # Add timeout/staleness metadata when applicable
            if bg_task.is_timed_out or bg_task.status.value == "timeout":
                metadata["is_timed_out"] = True
                metadata["timeout_configured"] = bg_task.timeout
                if bg_task.timed_out_at:
                    metadata["timed_out_at"] = bg_task.timed_out_at
                if bg_task.timeout_elapsed_seconds:
                    metadata["timeout_elapsed_seconds"] = bg_task.timeout_elapsed_seconds
            if hasattr(bg_task, "is_stale") and callable(bg_task.is_stale):
                # Check staleness with configurable threshold
                if bg_task.is_stale(self.config.deep_research_stale_task_seconds):
                    metadata["is_stale"] = True
                    metadata["last_activity"] = bg_task.last_activity
            # Include progress from persisted state if available
            if state:
                # Track status check count for polling mitigation
                state.status_check_count += 1
                state.last_status_check_at = datetime.now(timezone.utc)
                # Only persist for completed tasks; active tasks hold state in-memory
                # to avoid clobbering concurrent workflow saves (see comment at line 1750)
                # Use throttle logic to reduce disk I/O for frequent status checks
                if not is_active:
                    self._persist_state_if_needed(state)

                metadata.update(
                    {
                        "original_query": state.original_query,
                        "phase": state.phase.value,
                        "iteration": state.iteration,
                        "max_iterations": state.max_iterations,
                        "sub_queries_total": len(state.sub_queries),
                        "sub_queries_completed": len(state.completed_sub_queries()),
                        "source_count": len(state.sources),
                        "finding_count": len(state.findings),
                        "gap_count": len(state.unresolved_gaps()),
                        "total_tokens_used": state.total_tokens_used,
                        "is_failed": bool(state.metadata.get("failed")),
                        "failure_error": state.metadata.get("failure_error"),
                        "status_check_count": state.status_check_count,
                        "last_heartbeat_at": state.last_heartbeat_at.isoformat() if state.last_heartbeat_at else None,
                    }
                )
                # Build detailed status content when state is available
                status_lines = [
                    f"Research ID: {state.id}",
                    f"Query: {state.original_query}",
                    f"Task Status: {bg_task.status.value}",
                    f"Phase: {state.phase.value}",
                    f"Iteration: {state.iteration}/{state.max_iterations}",
                ]
                content = "\n".join(status_lines)
            else:
                content = f"Task status: {bg_task.status.value}"
            # Cleanup registries for completed tasks to prevent leaks.
            if not is_active:
                try:
                    self._cleanup_completed_task(research_id)
                    task_registry.remove(research_id)
                except Exception:
                    pass
            return WorkflowResult(
                success=True,
                content=content,
                metadata=metadata,
            )

        # Fall back to persisted state (task completed or not running)
        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        # Track status check count for polling mitigation
        state.status_check_count += 1
        state.last_status_check_at = datetime.now(timezone.utc)
        # Use throttle logic to reduce disk I/O for frequent status checks
        self._persist_state_if_needed(state)

        # Determine status string
        is_failed = bool(state.metadata.get("failed"))
        if is_failed:
            status_str = "Failed"
        elif state.completed_at:
            status_str = "Completed"
        else:
            status_str = "In Progress"

        status_lines = [
            f"Research ID: {state.id}",
            f"Query: {state.original_query}",
            f"Phase: {state.phase.value}",
            f"Iteration: {state.iteration}/{state.max_iterations}",
            f"Sub-queries: {len(state.completed_sub_queries())}/{len(state.sub_queries)} completed",
            f"Sources: {len(state.sources)} examined",
            f"Findings: {len(state.findings)}",
            f"Gaps: {len(state.unresolved_gaps())} unresolved",
            f"Status: {status_str}",
        ]
        if state.metadata.get("timeout"):
            status_lines.append("Timeout: True")
        if state.metadata.get("cancelled"):
            status_lines.append("Cancelled: True")
        if is_failed:
            failure_error = state.metadata.get("failure_error", "Unknown error")
            status_lines.append(f"Error: {failure_error}")

        # Build failed sub-queries list with reasons
        failed_sub_queries = [
            {
                "id": sq.id,
                "query": sq.query,
                "error": sq.error,
            }
            for sq in state.failed_sub_queries()
        ]

        return WorkflowResult(
            success=True,
            content="\n".join(status_lines),
            metadata={
                "research_id": state.id,
                "original_query": state.original_query,
                "phase": state.phase.value,
                "iteration": state.iteration,
                "max_iterations": state.max_iterations,
                "sub_queries_total": len(state.sub_queries),
                "sub_queries_completed": len(state.completed_sub_queries()),
                "sub_queries_failed": len(failed_sub_queries),
                "failed_sub_queries": failed_sub_queries,
                "source_count": len(state.sources),
                "finding_count": len(state.findings),
                "gap_count": len(state.unresolved_gaps()),
                "is_complete": state.completed_at is not None,
                "is_failed": is_failed,
                "failure_error": state.metadata.get("failure_error"),
                "total_tokens_used": state.total_tokens_used,
                "total_duration_ms": state.total_duration_ms,
                "timed_out": bool(state.metadata.get("timeout")),
                "cancelled": bool(state.metadata.get("cancelled")),
                "status_check_count": state.status_check_count,
                "last_heartbeat_at": state.last_heartbeat_at.isoformat() if state.last_heartbeat_at else None,
            },
        )

    def _get_report(self, research_id: Optional[str]) -> WorkflowResult:
        """Get the final report from a research session."""
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required",
            )

        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        if not state.report:
            return WorkflowResult(
                success=False,
                content="",
                error="Research report not yet generated",
            )

        # Build warnings list from allocation metadata
        warnings: list[str] = []
        allocation_meta = state.content_allocation_metadata or {}

        # Add warning if content was dropped
        if state.dropped_content_ids:
            warnings.append(f"Content truncated: {len(state.dropped_content_ids)} source(s) dropped for context limits")

        # Add warning if fidelity is degraded
        fidelity_level = allocation_meta.get("overall_fidelity_level") or ""
        if fidelity_level not in ("full", ""):
            warnings.append(f"Content fidelity: {fidelity_level} (some sources may be summarized)")

        # Add any warnings from allocation metadata
        if allocation_meta.get("warnings"):
            warnings.extend(allocation_meta["warnings"])

        return WorkflowResult(
            success=True,
            content=state.report,
            metadata={
                "research_id": state.id,
                "original_query": state.original_query,
                "source_count": len(state.sources),
                "finding_count": len(state.findings),
                "iteration": state.iteration,
                "is_complete": state.completed_at is not None,
                # Token management metadata
                "content_fidelity_schema_version": "1.0",
                "content_fidelity": state.content_fidelity,
                "dropped_content_ids": state.dropped_content_ids,
                "content_allocation_summary": {
                    "tokens_used": allocation_meta.get("tokens_used"),
                    "tokens_budget": allocation_meta.get("tokens_budget"),
                    "fidelity_score": allocation_meta.get("fidelity"),
                    "items_allocated": allocation_meta.get("items_allocated"),
                    "items_dropped": allocation_meta.get("items_dropped"),
                },
                "warnings": warnings,
            },
        )

    def _cancel_research(self, research_id: Optional[str]) -> WorkflowResult:
        """Cancel a running research task."""
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required",
            )

        bg_task = self.get_background_task(research_id)
        if bg_task is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"No running task found for '{research_id}'",
            )

        if bg_task.cancel():
            state = self.memory.load_deep_research(research_id)
            if state:
                state.mark_cancelled(phase_state=f"phase={state.phase.value}, iteration={state.iteration}")
                self.memory.save_deep_research(state)
                self._write_audit_event(
                    state,
                    "workflow_cancelled",
                    data={
                        "cancelled": True,
                        "terminal_status": "cancelled",
                    },
                    level="warning",
                )
            return WorkflowResult(
                success=True,
                content=f"Research '{research_id}' cancelled",
                metadata={"research_id": research_id, "cancelled": True},
            )
        else:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Task '{research_id}' already completed",
            )

    def _evaluate_research(self, research_id: Optional[str]) -> WorkflowResult:
        """Evaluate a completed research report using LLM-as-judge.

        Loads the research session, validates the report exists, runs
        evaluation across 6 quality dimensions, and returns scores.

        Args:
            research_id: ID of the research session to evaluate

        Returns:
            WorkflowResult with evaluation scores or error
        """
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required for evaluation",
            )

        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        if not state.report:
            return WorkflowResult(
                success=False,
                content="",
                error="Research report not yet generated. Complete research first.",
            )

        # Resolve evaluation provider/model/timeout from config
        eval_provider = getattr(self.config, "deep_research_evaluation_provider", None)
        eval_model = getattr(self.config, "deep_research_evaluation_model", None)
        eval_timeout = getattr(self.config, "deep_research_evaluation_timeout", 360.0)

        from foundry_mcp.core.research.evaluation.evaluator import evaluate_report

        coro = evaluate_report(
            workflow=self,
            state=state,
            provider_id=eval_provider,
            model=eval_model,
            timeout=eval_timeout,
        )

        # Run evaluation — dispatch onto the existing event loop when one is
        # running (MCP context) instead of spawning a separate loop in a
        # thread, which breaks cancellation and structured concurrency.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Called from within an async context — schedule onto the
            # running loop and block until done (safe because we're in a
            # worker thread, not the event loop thread).
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            result = future.result(timeout=eval_timeout + 30)
        else:
            result = asyncio.run(coro)

        # If evaluate_report returned a WorkflowResult (error), pass through
        if isinstance(result, WorkflowResult):
            return result

        # Build response from EvaluationResult
        score_summary = "\n".join(
            f"  {ds.name}: {ds.raw_score}/5 ({ds.normalized_score:.2f}) — {ds.rationale}"
            for ds in result.dimension_scores
        )
        content = (
            f"Evaluation of research '{research_id}':\n"
            f"Composite Score: {result.composite_score:.3f} (0-1 scale)\n\n"
            f"Dimension Scores:\n{score_summary}"
        )

        return WorkflowResult(
            success=True,
            content=content,
            metadata={
                "research_id": research_id,
                "evaluation": result.to_dict(),
                "composite_score": result.composite_score,
            },
        )

    def _run_sync(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
        task_timeout: Optional[float],
    ) -> WorkflowResult:
        """Run workflow synchronously with optional timeout enforcement.

        Uses a two-way dispatch: if an event loop is already running
        (e.g. MCP async context), offloads to a ThreadPoolExecutor;
        otherwise calls asyncio.run() directly.

        When task_timeout is set, wraps the coroutine in
        asyncio.wait_for() so the workflow is cancelled on timeout.
        """

        async def _run_with_timeout() -> WorkflowResult:
            coro = self._execute_workflow_async(
                state=state,
                provider_id=provider_id,
                timeout_per_operation=timeout_per_operation,
                max_concurrent=max_concurrent,
            )
            if task_timeout:
                return await asyncio.wait_for(coro, timeout=task_timeout)
            return await coro

        try:
            try:
                asyncio.get_running_loop()
                # Already in async context — run in a thread with a new loop
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _run_with_timeout())
                    return future.result()
            except RuntimeError:
                # No running loop — safe to call asyncio.run()
                return asyncio.run(_run_with_timeout())
        except asyncio.TimeoutError:
            timeout_message = f"Research timed out after {task_timeout}s"
            state.metadata["timeout"] = True
            state.metadata["abort_phase"] = state.phase.value
            state.metadata["abort_iteration"] = state.iteration
            state.mark_failed(timeout_message)
            self.memory.save_deep_research(state)
            self._write_audit_event(
                state,
                "workflow_timeout",
                data={
                    "timeout_seconds": task_timeout,
                    "abort_phase": state.phase.value,
                    "abort_iteration": state.iteration,
                },
                level="warning",
            )
            return WorkflowResult(
                success=False,
                content="",
                error=timeout_message,
                metadata={"research_id": state.id, "timeout": True},
            )
