"""Background task management mixin for DeepResearchWorkflow.

Handles starting, monitoring, and cleaning up background research tasks
that run in daemon threads.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from typing import TYPE_CHECKING, Optional

from foundry_mcp.core.background_task import BackgroundTask, TaskStatus
from foundry_mcp.core import task_registry
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.infrastructure import (
    _active_research_sessions,
    _active_sessions_lock,
)

if TYPE_CHECKING:
    from foundry_mcp.core.research.workflows.deep_research.core import (
        DeepResearchWorkflow,
    )

logger = logging.getLogger(__name__)


class BackgroundTaskMixin:
    """Background task management methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - _tasks, _tasks_lock (class-level task registry)
    - _execute_workflow_async(), _write_audit_event(), _record_workflow_error(),
      _flush_state() (cross-cutting methods)
    - memory (inherited from ResearchWorkflowBase)
    """

    def _start_background_task(
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
        task_timeout: Optional[float],
    ) -> WorkflowResult:
        """Start research as a background task using a daemon thread.

        Returns immediately with research_id. The actual workflow
        runs in a daemon thread using asyncio.run().

        This approach works correctly from sync MCP tool handlers where
        there is no running event loop.
        """
        # Create BackgroundTask tracking structure first
        bg_task = BackgroundTask(
            research_id=state.id,
            timeout=task_timeout,
        )
        with self._tasks_lock:
            self._tasks[state.id] = bg_task
        # Also register with global task registry for watchdog monitoring
        task_registry.register(bg_task)

        # Register session for crash handler visibility (under lock)
        with _active_sessions_lock:
            _active_research_sessions[state.id] = state

        # Reference to self for use in thread
        workflow = self

        def run_in_thread() -> None:
            """Thread target that runs the async workflow."""
            try:
                async def run_workflow() -> WorkflowResult:
                    """Execute the full workflow asynchronously."""
                    try:
                        coro = workflow._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        )
                        if task_timeout:
                            return await asyncio.wait_for(coro, timeout=task_timeout)
                        return await coro
                    except asyncio.CancelledError:
                        state.metadata["cancelled"] = True
                        workflow.memory.save_deep_research(state)
                        workflow._write_audit_event(
                            state,
                            "workflow_cancelled",
                            data={"cancelled": True},
                            level="warning",
                        )
                        return WorkflowResult(
                            success=False,
                            content="",
                            error="Research was cancelled",
                            metadata={"research_id": state.id, "cancelled": True},
                        )
                    except asyncio.TimeoutError:
                        timeout_message = f"Research timed out after {task_timeout}s"
                        state.metadata["timeout"] = True
                        state.metadata["abort_phase"] = state.phase.value
                        state.metadata["abort_iteration"] = state.iteration
                        state.mark_failed(timeout_message)
                        workflow.memory.save_deep_research(state)
                        workflow._write_audit_event(
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
                    except Exception as exc:
                        logger.exception("Background workflow failed: %s", exc)
                        workflow._write_audit_event(
                            state,
                            "workflow_error",
                            data={"error": str(exc)},
                            level="error",
                        )
                        return WorkflowResult(
                            success=False,
                            content="",
                            error=str(exc),
                            metadata={"research_id": state.id},
                        )

                # Run the async workflow in a new event loop
                result = asyncio.run(run_workflow())

                # Handle completion
                if result.metadata and result.metadata.get("timeout"):
                    bg_task.mark_timeout()
                    bg_task.result = result
                    bg_task.error = result.error
                else:
                    # Use core BackgroundTask mark_completed signature
                    if result.success:
                        bg_task.mark_completed(result=result)
                    else:
                        bg_task.mark_completed(result=result, error=result.error)

            except Exception as exc:
                # Log the exception with full traceback
                logger.exception(
                    "Background task failed for research %s: %s",
                    state.id, exc
                )
                bg_task.status = TaskStatus.FAILED
                bg_task.error = str(exc)
                bg_task.completed_at = time.time()
                # Record to error store and audit (best effort)
                try:
                    workflow._record_workflow_error(exc, state, "background_task")
                    workflow._write_audit_event(
                        state,
                        "background_task_failed",
                        data={
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        level="error",
                    )
                except Exception:
                    pass  # Already logged above
            finally:
                # Unregister from active sessions (under lock)
                with _active_sessions_lock:
                    _active_research_sessions.pop(state.id, None)
                # Ensure final state is persisted for completed/cancelled/failed workflows
                try:
                    workflow._flush_state(state)
                except Exception:
                    pass
                # Remove completed task from registries to avoid leaks
                try:
                    workflow._cleanup_completed_task(state.id)
                    task_registry.remove(state.id)
                except Exception:
                    pass

        # Create and start the daemon thread
        thread = threading.Thread(
            target=run_in_thread,
            name=f"deep-research-{state.id[:8]}",
            daemon=True,  # Don't prevent process exit
        )
        bg_task.thread = thread

        self._write_audit_event(
            state,
            "background_task_started",
            data={
                "task_timeout": task_timeout,
                "timeout_per_operation": timeout_per_operation,
                "max_concurrent": max_concurrent,
                "thread_name": thread.name,
            },
        )

        thread.start()

        return WorkflowResult(
            success=True,
            content=f"Research started in background: {state.id}",
            metadata={
                "research_id": state.id,
                "background": True,
                "phase": state.phase.value,
            },
        )

    def get_background_task(self: DeepResearchWorkflow, research_id: str) -> Optional[BackgroundTask]:
        """Get a background task by research ID."""
        with self._tasks_lock:
            return self._tasks.get(research_id)

    def _cleanup_completed_task(self: DeepResearchWorkflow, research_id: str) -> None:
        """Remove a completed task from the registry to free memory.

        Called when a background task finishes (success, failure, or timeout).
        """
        with self._tasks_lock:
            self._tasks.pop(research_id, None)

    @classmethod
    def cleanup_stale_tasks(cls, max_age_seconds: float = 3600) -> int:
        """Remove old completed tasks from the registry.

        This can be called periodically to clean up memory from completed tasks
        that haven't been explicitly cleaned up.

        Args:
            max_age_seconds: Maximum age in seconds for completed tasks (default 1 hour)

        Returns:
            Number of tasks removed
        """
        import time
        now = time.time()
        removed = 0
        with cls._tasks_lock:
            stale_ids = [
                task_id
                for task_id, task in cls._tasks.items()
                if task.is_done and task.completed_at
                and (now - task.completed_at) > max_age_seconds
            ]
            for task_id in stale_ids:
                del cls._tasks[task_id]
                removed += 1
        return removed
