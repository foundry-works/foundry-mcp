"""Deep Research workflow with async background execution.

Provides multi-phase iterative research through query decomposition,
parallel source gathering, content analysis, and synthesized reporting.

Key Features:
- Background execution via daemon threads with asyncio.run()
- Immediate research_id return on start
- Status polling while running
- Task lifecycle tracking with cancellation support
- Multi-agent supervisor orchestration hooks

Note: Uses daemon threads (not asyncio.create_task()) to ensure background
execution works correctly from synchronous MCP tool handlers where there
is no running event loop.

Inspired by:
- open_deep_research: Multi-agent supervision with think-tool pauses
- Claude-Deep-Research: Dual-source search with link following
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.background_task import BackgroundTask
from foundry_mcp.core import task_registry
from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    DeepResearchPhase,
    DeepResearchState,
    DOMAIN_TIERS,
    ResearchMode,
    ResearchSource,
)
from foundry_mcp.core.error_collection import ErrorRecord
from foundry_mcp.core.error_store import FileErrorStore
from foundry_mcp.core.research.providers import SearchProvider
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
    _active_research_sessions,
    _active_sessions_lock,
    _active_research_memory,
    _persist_active_sessions,
    _crash_handler,
    _cleanup_on_exit,
    install_crash_handler,
)

from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_hostname,
)
from foundry_mcp.core.research.workflows.deep_research.orchestration import (
    AgentRole,
    PHASE_TO_AGENT,
    AgentDecision,
    SupervisorHooks,
    SupervisorOrchestrator,
)
from foundry_mcp.core.research.workflows.deep_research.phases.planning import (
    PlanningPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.gathering import (
    GatheringPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.analysis import (
    AnalysisPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    SynthesisPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.refinement import (
    RefinementPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.background_tasks import (
    BackgroundTaskMixin,
)
from foundry_mcp.core.research.workflows.deep_research.session_management import (
    SessionManagementMixin,
)

logger = logging.getLogger(__name__)

# Install crash handler on import (matches original side-effect behavior)
install_crash_handler()


# =============================================================================
# Deep Research Workflow
# =============================================================================


class DeepResearchWorkflow(PlanningPhaseMixin, GatheringPhaseMixin, AnalysisPhaseMixin, SynthesisPhaseMixin, RefinementPhaseMixin, BackgroundTaskMixin, SessionManagementMixin, ResearchWorkflowBase):
    """Multi-phase deep research workflow with background execution.

    Supports:
    - Async execution with immediate research_id return
    - Status polling while research runs in background
    - Cancellation and timeout handling
    - Multi-agent supervisor hooks
    - Session persistence for resume capability

    Workflow Phases:
    1. PLANNING - Decompose query into sub-queries
    2. GATHERING - Execute sub-queries in parallel
    3. ANALYSIS - Extract findings and assess quality
    4. SYNTHESIS - Generate comprehensive report
    5. REFINEMENT - Identify gaps and iterate if needed
    """

    # Class-level task registry for background task tracking
    # Uses regular dict (not WeakValueDictionary) to prevent tasks from being GC'd while running
    # Protected by _tasks_lock for thread safety
    _tasks: dict[str, BackgroundTask] = {}
    _tasks_lock = threading.Lock()

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
        hooks: Optional[SupervisorHooks] = None,
    ) -> None:
        """Initialize deep research workflow.

        Args:
            config: Research configuration
            memory: Optional memory instance for persistence
            hooks: Optional supervisor hooks for orchestration
        """
        super().__init__(config, memory)
        global _active_research_memory
        _active_research_memory = self.memory
        self.hooks = hooks or SupervisorHooks()
        self.orchestrator = SupervisorOrchestrator()
        self._search_providers: dict[str, SearchProvider] = {}
        # Track last persistence time for throttling (see status_persistence_throttle_seconds)
        self._last_persisted_at: datetime | None = None
        # Track last persisted phase/iteration for change detection
        self._last_persisted_phase: DeepResearchPhase | None = None
        self._last_persisted_iteration: int | None = None

    def _audit_enabled(self) -> bool:
        """Return True if audit artifacts are enabled."""
        return bool(getattr(self.config, "deep_research_audit_artifacts", True))

    def _sync_persistence_tracking_from_state(self, state: DeepResearchState) -> None:
        """Sync persistence tracking fields from state metadata if available.

        This ensures throttling works across workflow instances by loading
        the last persisted timestamp/phase/iteration from persisted state.
        """
        if (
            self._last_persisted_at is not None
            and self._last_persisted_phase is not None
            and self._last_persisted_iteration is not None
        ):
            return

        meta = state.metadata.get("_status_persistence")
        if not isinstance(meta, dict):
            return

        # Load last persisted timestamp
        if self._last_persisted_at is None:
            raw_ts = meta.get("last_persisted_at")
            if isinstance(raw_ts, datetime):
                ts = raw_ts
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                self._last_persisted_at = ts
            elif isinstance(raw_ts, str):
                try:
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    self._last_persisted_at = ts
                except ValueError:
                    pass

        # Load last persisted phase
        if self._last_persisted_phase is None:
            raw_phase = meta.get("last_persisted_phase")
            if isinstance(raw_phase, DeepResearchPhase):
                self._last_persisted_phase = raw_phase
            elif isinstance(raw_phase, str):
                try:
                    self._last_persisted_phase = DeepResearchPhase(raw_phase)
                except ValueError:
                    pass

        # Load last persisted iteration
        if self._last_persisted_iteration is None:
            raw_iter = meta.get("last_persisted_iteration")
            if isinstance(raw_iter, int):
                self._last_persisted_iteration = raw_iter

    def _is_terminal_state(self, state: DeepResearchState) -> bool:
        """Check if state represents a terminal condition (completed or failed)."""
        if state.completed_at is not None:
            return True
        if state.metadata.get("failed"):
            return True
        return False

    def _should_persist_status(self, state: DeepResearchState) -> bool:
        """Determine if state should be persisted based on throttle rules.

        Priority (highest to lowest):
        1. Terminal state (completed/failed) - always persist
        2. Phase/iteration change - always persist
        3. Throttle interval elapsed - persist if interval exceeded

        A throttle_seconds of 0 means always persist (current behavior).

        Args:
            state: Current deep research state

        Returns:
            True if state should be persisted, False to skip
        """
        # Sync persisted tracking fields from state metadata if needed
        self._sync_persistence_tracking_from_state(state)

        # Priority 1: Terminal states always persist
        if self._is_terminal_state(state):
            return True

        # Priority 2: Phase or iteration change always persists
        if (
            self._last_persisted_phase is not None
            and state.phase != self._last_persisted_phase
        ):
            return True
        if (
            self._last_persisted_iteration is not None
            and state.iteration != self._last_persisted_iteration
        ):
            return True

        # Priority 3: Check throttle interval
        throttle_seconds = getattr(
            self.config, "status_persistence_throttle_seconds", 5
        )

        # 0 means always persist (backwards compatibility)
        if throttle_seconds == 0:
            return True

        # No previous persistence - should persist
        if self._last_persisted_at is None:
            return True

        # Check if throttle interval has elapsed
        elapsed = (datetime.now(timezone.utc) - self._last_persisted_at).total_seconds()
        return elapsed >= throttle_seconds

    def _persist_state(self, state: DeepResearchState) -> None:
        """Persist state and update tracking fields.

        Updates _last_persisted_at, _last_persisted_phase, and
        _last_persisted_iteration after successful save.

        Args:
            state: State to persist
        """
        now = datetime.now(timezone.utc)
        state.metadata["_status_persistence"] = {
            "last_persisted_at": now.isoformat(),
            "last_persisted_phase": state.phase.value,
            "last_persisted_iteration": state.iteration,
        }
        self.memory.save_deep_research(state)
        logger.debug(
            "Status persisted: research_id=%s phase=%s iteration=%d",
            state.id,
            state.phase.value,
            state.iteration,
        )
        self._last_persisted_at = now
        self._last_persisted_phase = state.phase
        self._last_persisted_iteration = state.iteration

    def _persist_state_if_needed(self, state: DeepResearchState) -> bool:
        """Conditionally persist state based on throttle rules.

        Args:
            state: State to potentially persist

        Returns:
            True if state was persisted, False if skipped
        """
        if self._should_persist_status(state):
            try:
                self._persist_state(state)
                return True
            except Exception as exc:
                logger.debug("Failed to persist state: %s", exc)
                return False
        logger.debug(
            "Status persistence skipped (throttled): research_id=%s phase=%s iteration=%d",
            state.id,
            state.phase.value,
            state.iteration,
        )
        return False

    def _flush_state(self, state: DeepResearchState) -> None:
        """Force-persist state, bypassing throttle rules.

        Use this for workflow completion paths (success, failure, cancellation)
        to ensure final state is always saved regardless of throttle interval.

        This guarantees:
        - Token usage/cache data is persisted
        - Final status is captured
        - Completion timestamp is saved

        Args:
            state: State to persist
        """
        self._persist_state(state)

    def _audit_path(self, research_id: str) -> Path:
        """Resolve audit artifact path for a research session."""
        # Use memory's base_path which is set from ServerConfig.get_research_dir()
        return self.memory.base_path / "deep_research" / f"{research_id}.audit.jsonl"

    def _prepare_audit_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare audit payload based on configured verbosity level.

        In 'full' mode: Returns data unchanged for complete audit trail.
        In 'minimal' mode: Sets large text fields to null while preserving
        metrics and schema shape for analysis compatibility.

        Nulled fields in minimal mode:
        - Top-level: system_prompt, user_prompt, raw_response, report, error, traceback
        - Nested: findings[*].content, gaps[*].description

        Preserved fields (always included):
        - provider_id, model_used, tokens_used, duration_ms
        - sources_added, report_length, parse_success
        - All other scalar metrics

        Args:
            data: Original audit event data dictionary

        Returns:
            Processed data dictionary (same schema shape, potentially nulled values)
        """
        verbosity = self.config.audit_verbosity

        # Full mode: return unchanged
        if verbosity == "full":
            return data

        # Minimal mode: null out large text fields while preserving schema
        result = dict(data)  # Shallow copy

        # Top-level fields to null
        text_fields = {
            "system_prompt",
            "user_prompt",
            "raw_response",
            "report",
            "error",
            "traceback",
        }
        for field in text_fields:
            if field in result:
                result[field] = None

        # Handle nested findings array
        if "findings" in result and isinstance(result["findings"], list):
            result["findings"] = [
                {**f, "content": None} if isinstance(f, dict) and "content" in f else f
                for f in result["findings"]
            ]

        # Handle nested gaps array
        if "gaps" in result and isinstance(result["gaps"], list):
            result["gaps"] = [
                {**g, "description": None} if isinstance(g, dict) and "description" in g else g
                for g in result["gaps"]
            ]

        return result

    def _write_audit_event(
        self,
        state: Optional[DeepResearchState],
        event_type: str,
        data: Optional[dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """Write a JSONL audit event for deep research observability."""
        if not self._audit_enabled():
            return

        research_id = state.id if state else None
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_id": uuid4().hex,
            "event_type": event_type,
            "level": level,
            "research_id": research_id,
            "phase": state.phase.value if state else None,
            "iteration": state.iteration if state else None,
            "data": self._prepare_audit_payload(data or {}),
        }

        try:
            if research_id is None:
                return
            path = self._audit_path(research_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
        except Exception as exc:
            logger.error("Failed to write audit event: %s", exc)
            # Fallback to stderr for crash visibility
            print(
                f"AUDIT_FALLBACK: {event_type} for {research_id} - {exc}",
                file=sys.stderr,
                flush=True,
            )

    # Search provider config methods provided by GatheringPhaseMixin (phases/gathering.py)

    def _record_workflow_error(
        self,
        error: Exception,
        state: DeepResearchState,
        context: str,
    ) -> None:
        """Record error to the persistent error store.

        Args:
            error: The exception that occurred
            state: Current research state
            context: Context string (e.g., "background_task", "orchestrator")
        """
        try:
            error_store = FileErrorStore(Path.home() / ".foundry-mcp" / "errors")
            record = ErrorRecord(
                id=f"err_{uuid4().hex[:12]}",
                fingerprint=f"deep-research:{context}:{type(error).__name__}",
                error_code="WORKFLOW_ERROR",
                error_type="internal",
                tool_name=f"deep-research:{context}",
                correlation_id=state.id,
                message=str(error),
                exception_type=type(error).__name__,
                stack_trace=traceback.format_exc(),
                input_summary={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
            )
            error_store.append(record)
        except Exception as store_err:
            logger.error("Failed to record error to store: %s", store_err)

    def _safe_orchestrator_transition(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> None:
        """Safely execute orchestrator phase transition with error logging.

        This wraps orchestrator calls with exception handling to ensure any
        failures are properly logged and recorded before re-raising.

        Args:
            state: Current research state
            phase: The phase that just completed

        Raises:
            Exception: Re-raises any exception after logging
        """
        try:
            self.orchestrator.evaluate_phase_completion(state, phase)
            prompt = self.orchestrator.get_reflection_prompt(state, phase)
            self.hooks.think_pause(state, prompt)
            self.orchestrator.record_to_state(state)
            state.advance_phase()
        except Exception as exc:
            logger.exception(
                "Orchestrator transition failed for phase %s, research %s: %s",
                phase.value,
                state.id,
                exc,
            )
            self._write_audit_event(
                state,
                "orchestrator_error",
                data={
                    "phase": phase.value,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                level="error",
            )
            self._record_workflow_error(exc, state, f"orchestrator_{phase.value}")
            raise  # Re-raise to be caught by workflow exception handler

    # =========================================================================
    # Public API
    # =========================================================================

    def execute(
        self,
        query: Optional[str] = None,
        research_id: Optional[str] = None,
        action: str = "start",
        provider_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 3,
        max_sub_queries: int = 5,
        max_sources_per_query: int = 5,
        follow_links: bool = True,
        timeout_per_operation: float = 120.0,
        max_concurrent: int = 3,
        background: bool = False,
        task_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute deep research workflow.

        Actions:
        - start: Begin new research session
        - continue: Resume existing session
        - status: Get current status
        - report: Get final report
        - cancel: Cancel running task

        Args:
            query: Research query (required for 'start')
            research_id: Session ID (required for continue/status/report/cancel)
            action: One of 'start', 'continue', 'status', 'report', 'cancel'
            provider_id: Provider for LLM operations
            system_prompt: Optional custom system prompt
            max_iterations: Maximum refinement iterations (default: 3)
            max_sub_queries: Maximum sub-queries to generate (default: 5)
            max_sources_per_query: Maximum sources per query (default: 5)
            follow_links: Whether to extract content from URLs (default: True)
            timeout_per_operation: Timeout per operation in seconds (default: 30)
            max_concurrent: Maximum concurrent operations (default: 3)
            background: Run in background, return immediately (default: False)
            task_timeout: Overall timeout for background task (optional)

        Returns:
            WorkflowResult with research state or error
        """
        try:
            if action == "start":
                return self._start_research(
                    query=query,
                    provider_id=provider_id,
                    system_prompt=system_prompt,
                    max_iterations=max_iterations,
                    max_sub_queries=max_sub_queries,
                    max_sources_per_query=max_sources_per_query,
                    follow_links=follow_links,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                    background=background,
                    task_timeout=task_timeout,
                )
            elif action == "continue":
                return self._continue_research(
                    research_id=research_id,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                    background=background,
                    task_timeout=task_timeout,
                )
            elif action == "status":
                return self._get_status(research_id=research_id)
            elif action == "report":
                return self._get_report(research_id=research_id)
            elif action == "cancel":
                return self._cancel_research(research_id=research_id)
            else:
                return WorkflowResult(
                    success=False,
                    content="",
                    error=f"Unknown action '{action}'. Use: start, continue, status, report, cancel",
                )
        except Exception as exc:
            # Catch all exceptions to ensure graceful failure
            logger.exception("Deep research execute failed for action '%s': %s", action, exc)
            return WorkflowResult(
                success=False,
                content="",
                error=f"Deep research failed: {exc}",
                metadata={
                    "action": action,
                    "research_id": research_id,
                    "error_type": exc.__class__.__name__,
                },
            )

    # Background task methods provided by BackgroundTaskMixin (background_tasks.py)

    # =========================================================================
    # Action Handlers
    # =========================================================================

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

        # Resolve per-phase providers and models from config
        # Supports ProviderSpec format: "[cli]gemini:pro" -> (provider_id, model)
        planning_pid, planning_model = self.config.resolve_phase_provider("planning")
        analysis_pid, analysis_model = self.config.resolve_phase_provider("analysis")
        synthesis_pid, synthesis_model = self.config.resolve_phase_provider("synthesis")
        refinement_pid, refinement_model = self.config.resolve_phase_provider("refinement")

        # Create initial state with per-phase provider configuration
        state = DeepResearchState(
            original_query=query,
            max_iterations=max_iterations,
            max_sub_queries=max_sub_queries,
            max_sources_per_query=max_sources_per_query,
            follow_links=follow_links,
            research_mode=ResearchMode(self.config.deep_research_mode),
            system_prompt=system_prompt,
            # Per-phase providers: explicit provider_id overrides config
            planning_provider=provider_id or planning_pid,
            analysis_provider=provider_id or analysis_pid,
            synthesis_provider=provider_id or synthesis_pid,
            refinement_provider=provider_id or refinement_pid,
            # Per-phase models from ProviderSpec (only used if provider_id not overridden)
            planning_model=None if provider_id else planning_model,
            analysis_model=None if provider_id else analysis_model,
            synthesis_model=None if provider_id else synthesis_model,
            refinement_model=None if provider_id else refinement_model,
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

        # Synchronous execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, run directly
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._execute_workflow_async(
                    state=state,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
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

        # Continue from current phase synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._execute_workflow_async(
                    state=state,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
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
                # Check staleness with default threshold (300s)
                if bg_task.is_stale(300.0):
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

                metadata.update({
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
                })
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
            warnings.append(
                f"Content truncated: {len(state.dropped_content_ids)} source(s) dropped for context limits"
            )

        # Add warning if fidelity is degraded
        fidelity_level = allocation_meta.get("overall_fidelity_level") or ""
        if fidelity_level not in ("full", ""):
            warnings.append(
                f"Content fidelity: {fidelity_level} (some sources may be summarized)"
            )

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
                self._write_audit_event(
                    state,
                    "workflow_cancelled",
                    data={"cancelled": True},
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

    # =========================================================================
    # Async Workflow Execution
    # =========================================================================

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
        """Execute common phase lifecycle: cancel → timer → hooks → audit → execute → error → hooks → audit → transition.

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
                    "total_input_tokens": sum(
                        m.input_tokens for m in state.phase_metrics
                    ),
                    "total_output_tokens": sum(
                        m.output_tokens for m in state.phase_metrics
                    ),
                    "total_cached_tokens": sum(
                        m.cached_tokens for m in state.phase_metrics
                    ),
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
                        set(
                            h
                            for s in state.sources
                            if s.url and (h := _extract_hostname(s.url))
                        )
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
                    if hasattr(provider, 'aclose'):
                        await provider.aclose()
                    elif hasattr(provider, 'close'):
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

    # =========================================================================
    # Phase Implementations
    # =========================================================================
    # Planning phase methods provided by PlanningPhaseMixin (phases/planning.py)

    # Gathering phase methods provided by GatheringPhaseMixin (phases/gathering.py)
    # Analysis phase methods provided by AnalysisPhaseMixin (phases/analysis.py)
    # Synthesis phase methods provided by SynthesisPhaseMixin (phases/synthesis.py)
    # Refinement phase methods provided by RefinementPhaseMixin (phases/refinement.py)

    # Session management methods provided by SessionManagementMixin (session_management.py)

