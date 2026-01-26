"""Background task lifecycle management with cooperative cancellation.

Provides centralized task tracking for background research operations,
supporting both thread-based and asyncio-based execution modes with
unified cancellation, timeout handling, and status tracking.

Key Features:
- Dual cancellation support: threading.Event for threads, asyncio.Event for tasks
- Unified timeout handling and elapsed time tracking
- Status lifecycle: PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED/TIMEOUT
- Type-agnostic result storage for both asyncio and thread execution
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a background task.

    Attributes:
        PENDING: Task created but not started.
        RUNNING: Task currently executing.
        COMPLETED: Task finished successfully.
        FAILED: Task finished with error.
        CANCELLED: Task cancelled by user.
        TIMEOUT: Task exceeded timeout limit.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class BackgroundTask:
    """Tracks a background task with lifecycle management and cancellation support.

    Supports both asyncio.Task-based and thread-based execution for unified
    lifecycle management, cooperative cancellation, and timeout handling.

    Attributes:
        research_id: Unique identifier for the task/research session.
        task: Optional asyncio task running the workflow.
        thread: Optional thread running the workflow.
        timeout: Optional timeout in seconds.
        status: Current task status from TaskStatus enum.
        started_at: Unix timestamp when task was created.
        completed_at: Unix timestamp when task completed (None if running).
        error: Error message if task failed (None if successful/running).
        result: Result object from task completion (None if running/failed).
    """

    def __init__(
        self,
        research_id: str,
        task: Optional[asyncio.Task[Any]] = None,
        thread: Optional[threading.Thread] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize background task.

        Args:
            research_id: ID of the research session or task.
            task: Optional asyncio task running the workflow.
            thread: Optional thread running the workflow (preferred for MCP handlers).
            timeout: Optional timeout in seconds. None means no timeout.

        Raises:
            ValueError: If both task and thread are provided.
        """
        self.research_id = research_id
        self.task = task
        self.thread = thread
        self.timeout = timeout
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[Any] = None
        self.last_activity: float = time.time()  # Track last activity for staleness
        self.timed_out_at: Optional[float] = None  # Timestamp when timeout was detected
        self.timeout_elapsed_seconds: Optional[float] = None  # Elapsed time at timeout

        # Event for signaling cancellation to thread-based execution
        self._cancel_event = threading.Event()

        # Event for signaling cancellation to asyncio-based execution
        self._async_cancel_event = asyncio.Event()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time since task start in milliseconds.
        """
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000

    @property
    def is_timed_out(self) -> bool:
        """Check if task has exceeded timeout.

        Returns:
            True if timeout is set and exceeded, False otherwise.
        """
        if self.timed_out_at is not None or self.status == TaskStatus.TIMEOUT:
            return True
        if self.completed_at is not None:
            return False
        if self.timeout is None:
            return False
        return (time.time() - self.started_at) > self.timeout

    def is_stale(self, stale_threshold: float = 300.0) -> bool:
        """Check if task has been inactive beyond the staleness threshold.

        A task is considered stale if it is still running but has not
        reported any activity (via touch()) for longer than the threshold.

        Args:
            stale_threshold: Seconds of inactivity before task is stale (default 300).

        Returns:
            True if task is running and inactive beyond threshold, False otherwise.
        """
        if self.status != TaskStatus.RUNNING:
            return False
        return (time.time() - self.last_activity) > stale_threshold

    def touch(self) -> None:
        """Update last_activity timestamp to indicate progress.

        Call this method periodically from long-running tasks to indicate
        they are still making progress and should not be marked as stale.
        """
        self.last_activity = time.time()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancellation event is set, False otherwise.
        """
        return self._cancel_event.is_set()

    @property
    def is_async_cancelled(self) -> bool:
        """Check if asyncio cancellation has been requested.

        Returns:
            True if asyncio cancellation event is set, False otherwise.
        """
        return self._async_cancel_event.is_set()

    @property
    def is_done(self) -> bool:
        """Check if the task is done (for both thread and asyncio modes).

        Returns:
            True if the task has completed, False if still running.
        """
        if self.thread is not None:
            return not self.thread.is_alive()
        elif self.task is not None:
            return self.task.done()
        # Neither thread nor task - consider done (shouldn't happen in practice)
        return True

    def cancel(self, timeout: float = 5.0) -> bool:
        """Two-phase cancellation: cooperative then forced.

        Phase 1: Set cancellation event and wait for cooperative shutdown.
        Phase 2: Force cancel if still running after timeout.

        This method implements graceful shutdown by first signaling the task
        to stop via cancellation events, then forcibly cancelling if needed.

        Args:
            timeout: Time in seconds to wait for cooperative shutdown (default 5.0).
                     Use 0 to skip to immediate forced cancellation.

        Returns:
            True if cancellation was requested, False if already done.
        """
        # Handle thread-based execution
        if self.thread is not None:
            if not self.thread.is_alive():
                return False

            # Phase 1: Signal cancellation and wait for cooperative shutdown
            logger.debug(
                "Cancellation phase 1 (cooperative): signaling thread %s to stop",
                self.thread.name,
            )
            self._cancel_event.set()
            self.thread.join(timeout=timeout)

            # Phase 2: Check if thread is still alive
            if self.thread.is_alive():
                logger.warning(
                    "Thread %s did not stop cooperatively after %.1fs, "
                    "cannot force kill in Python",
                    self.thread.name,
                    timeout,
                )
            else:
                logger.debug(
                    "Thread %s stopped cooperatively within %.1fs",
                    self.thread.name,
                    timeout,
                )

            self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()
            return True

        # Handle asyncio-based execution
        elif self.task is not None:
            if self.task.done():
                return False

            # Phase 1: Signal asyncio cancellation
            logger.debug("Cancellation phase 1 (cooperative): signaling asyncio task to stop")
            self._async_cancel_event.set()

            # Phase 2: Force cancel the task
            if not self.task.done():
                logger.debug(
                    "Cancellation phase 2 (forced): calling task.cancel() on asyncio task"
                )
                self.task.cancel()

            self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()
            return True

        return False

    def force_cancel(self) -> bool:
        """Force cancel the task, bypassing cooperative shutdown.

        This method immediately forces cancellation without waiting for
        cooperative shutdown. Use cancel() for graceful shutdown instead.

        For thread-based execution: Logs warning since threads cannot be
        forcibly killed in Python.
        For asyncio-based execution: Immediately calls task.cancel().

        Returns:
            True if cancellation was requested, False if already done.
        """
        # Handle thread-based execution
        if self.thread is not None:
            if not self.thread.is_alive():
                return False

            logger.warning(
                "Force cancel requested for thread %s, but threads cannot be forcibly "
                "killed in Python. Set cancellation event and hoping for cooperation.",
                self.thread.name,
            )
            self._cancel_event.set()
            self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()
            return True

        # Handle asyncio-based execution
        elif self.task is not None:
            if self.task.done():
                return False

            logger.debug("Force cancel: immediately cancelling asyncio task")
            self._async_cancel_event.set()
            self.task.cancel()
            self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()
            return True

        return False

    def mark_completed(self, result: Optional[Any] = None, error: Optional[str] = None) -> None:
        """Mark task as completed with result or error.

        Args:
            result: Result object from task completion (None if failed).
            error: Error message if task failed (None if successful).
        """
        if error is not None:
            self.status = TaskStatus.FAILED
            self.error = error
        else:
            self.status = TaskStatus.COMPLETED
            self.result = result

        self.completed_at = time.time()

    def mark_timeout(self) -> None:
        """Mark task as timed out with metadata persistence.

        Sets the task status to TIMEOUT and records:
        - timed_out_at: Timestamp when timeout was detected
        - timeout_elapsed_seconds: How long the task ran before timing out
        - completed_at: Completion timestamp (same as timed_out_at)
        """
        now = time.time()
        self.status = TaskStatus.TIMEOUT
        self.timed_out_at = now
        self.timeout_elapsed_seconds = now - self.started_at
        self.completed_at = now

    def cancel_event(self) -> threading.Event:
        """Get the threading cancel event for thread-based execution.

        Returns:
            threading.Event that signals cancellation to thread code.
        """
        return self._cancel_event

    async def async_cancel_event(self) -> asyncio.Event:
        """Get the asyncio cancel event for asyncio-based execution.

        Returns:
            asyncio.Event that signals cancellation to asyncio code.
        """
        return self._async_cancel_event
