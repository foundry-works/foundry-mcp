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

import logging
import threading
from typing import Any, Optional

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.background_task import BackgroundTask
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
)
from foundry_mcp.core.research.providers import SearchProvider
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.action_handlers import (
    ActionHandlersMixin,
)
from foundry_mcp.core.research.workflows.deep_research.audit import (
    AuditMixin,
)
from foundry_mcp.core.research.workflows.deep_research.background_tasks import (
    BackgroundTaskMixin,
)
from foundry_mcp.core.research.workflows.deep_research.error_handling import (
    ErrorHandlingMixin,
)
from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
    install_crash_handler,
)
from foundry_mcp.core.research.workflows.deep_research.orchestration import (
    SupervisorHooks,
    SupervisorOrchestrator,
)

# Extracted mixins
from foundry_mcp.core.research.workflows.deep_research.persistence import (
    PersistenceMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.analysis import (
    AnalysisPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.clarification import (
    ClarificationPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.gathering import (
    GatheringPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
    TopicResearchMixin,
)

# Phase mixins
from foundry_mcp.core.research.workflows.deep_research.phases.planning import (
    PlanningPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.refinement import (
    RefinementPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    SynthesisPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.session_management import (
    SessionManagementMixin,
)
from foundry_mcp.core.research.workflows.deep_research.workflow_execution import (
    WorkflowExecutionMixin,
)

logger = logging.getLogger(__name__)

# Install crash handler on import (matches original side-effect behavior)
install_crash_handler()


# =============================================================================
# Deep Research Workflow
# =============================================================================


class DeepResearchWorkflow(
    PersistenceMixin,
    AuditMixin,
    ErrorHandlingMixin,
    ActionHandlersMixin,
    WorkflowExecutionMixin,
    ClarificationPhaseMixin,
    PlanningPhaseMixin,
    GatheringPhaseMixin,
    TopicResearchMixin,
    AnalysisPhaseMixin,
    SynthesisPhaseMixin,
    RefinementPhaseMixin,
    BackgroundTaskMixin,
    SessionManagementMixin,
    ResearchWorkflowBase,
):
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
        self._last_persisted_at = None
        # Track last persisted phase/iteration for change detection
        self._last_persisted_phase: DeepResearchPhase | None = None
        self._last_persisted_iteration: int | None = None

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
