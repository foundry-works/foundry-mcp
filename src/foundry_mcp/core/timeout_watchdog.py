"""Timeout watchdog for background task monitoring.

Provides active detection and handling of timed-out background tasks
with proper state persistence and audit events. The watchdog runs as
a background asyncio task, periodically checking all registered tasks
for timeout and staleness conditions.

Key Features:
- Background polling loop with configurable interval (default 10s)
- Timeout detection via is_timed_out property on BackgroundTask
- Staleness detection for tasks that haven't made progress
- Automatic timeout cancellation trigger
- Audit event emission for timeout events
- Clean lifecycle management (start/stop)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from foundry_mcp.core.background_task import BackgroundTask

logger = logging.getLogger(__name__)


class TimeoutWatchdog:
    """Monitors background tasks for timeout and staleness conditions.

    The watchdog runs as a background asyncio task that periodically
    checks all registered tasks in the task registry. When a task
    exceeds its timeout or becomes stale, the watchdog triggers
    appropriate handling (cancellation, status updates, audit events).

    Attributes:
        poll_interval: Seconds between polling cycles (default 10).
        stale_threshold: Seconds of inactivity before task is stale (default 300).
        on_timeout: Optional callback invoked when a task times out.
        on_stale: Optional callback invoked when a task becomes stale.

    Example:
        watchdog = TimeoutWatchdog(poll_interval=10.0)
        await watchdog.start()
        # ... application runs ...
        await watchdog.stop()
    """

    def __init__(
        self,
        poll_interval: float = 10.0,
        stale_threshold: float = 300.0,
        on_timeout: Optional[Callable[["BackgroundTask"], None]] = None,
        on_stale: Optional[Callable[["BackgroundTask"], None]] = None,
    ) -> None:
        """Initialize the timeout watchdog.

        Args:
            poll_interval: Seconds between polling cycles (default 10).
            stale_threshold: Seconds without progress before task is stale (default 300).
            on_timeout: Optional callback when a task times out.
            on_stale: Optional callback when a task becomes stale.
        """
        self.poll_interval = poll_interval
        self.stale_threshold = stale_threshold
        self.on_timeout = on_timeout
        self.on_stale = on_stale

        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the watchdog is currently running.

        Returns:
            True if the watchdog polling loop is active.
        """
        return self._running and self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the watchdog background task.

        Creates and starts the background polling loop. If the watchdog
        is already running, this is a no-op.

        The watchdog will continue running until stop() is called.
        """
        if self.is_running:
            logger.debug("TimeoutWatchdog already running, ignoring start()")
            return

        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="timeout-watchdog")
        logger.info(
            "TimeoutWatchdog started with poll_interval=%.1fs, stale_threshold=%.1fs",
            self.poll_interval,
            self.stale_threshold,
        )

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the watchdog background task.

        Signals the polling loop to stop and waits for graceful shutdown.
        If the task doesn't stop within the timeout, it will be cancelled.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown (default 5).
        """
        if not self._running or self._task is None:
            logger.debug("TimeoutWatchdog not running, ignoring stop()")
            return

        logger.info("TimeoutWatchdog stopping...")
        self._stop_event.set()
        self._running = False

        try:
            await asyncio.wait_for(self._task, timeout=timeout)
            logger.info("TimeoutWatchdog stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning(
                "TimeoutWatchdog did not stop within %.1fs, cancelling", timeout
            )
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass

        self._task = None

    async def _poll_loop(self) -> None:
        """Main polling loop that checks tasks periodically.

        Runs until stop() is called. Each iteration:
        1. Gets all tasks from the registry
        2. Checks each running task for timeout/staleness
        3. Handles timed-out or stale tasks
        4. Sleeps for poll_interval seconds
        """
        logger.debug("TimeoutWatchdog poll loop started")

        while not self._stop_event.is_set():
            try:
                await self._check_tasks()
            except Exception as e:
                logger.exception("Error in watchdog poll loop: %s", e)

            # Wait for poll_interval or until stop is requested
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval
                )
                # If we get here, stop was requested
                break
            except asyncio.TimeoutError:
                # Normal case: poll interval elapsed, continue loop
                pass

        logger.debug("TimeoutWatchdog poll loop exited")

    async def _check_tasks(self) -> None:
        """Check all registered tasks for timeout and staleness.

        Iterates through all tasks in the registry and handles any
        that have timed out or become stale.
        """
        from foundry_mcp.core.background_task import TaskStatus
        from foundry_mcp.core.task_registry import get_task_registry_async

        registry = await get_task_registry_async()

        for _task_id, task in list(registry.items()):
            # Only check running tasks
            if task.status != TaskStatus.RUNNING:
                continue

            # Check for timeout
            if task.is_timed_out:
                await self._handle_timeout(task)
                continue

            # Check for staleness
            if task.is_stale(self.stale_threshold):
                await self._handle_stale(task)

    async def _handle_timeout(self, task: "BackgroundTask") -> None:
        """Handle a timed-out task.

        Triggers cancellation, marks the task as timed out, emits audit event,
        and invokes the on_timeout callback if configured.

        Args:
            task: The BackgroundTask that has timed out.
        """
        elapsed_seconds = task.elapsed_ms / 1000
        logger.warning(
            "Task %s timed out after %.1fs (timeout=%.1fs)",
            task.research_id,
            elapsed_seconds,
            task.timeout,
        )

        # Trigger cancellation of the underlying task/thread
        # Use force_cancel since the task has already exceeded its timeout
        try:
            task.force_cancel()
            logger.debug("Cancellation triggered for timed-out task %s", task.research_id)
        except Exception as e:
            logger.exception(
                "Error triggering cancellation for task %s: %s", task.research_id, e
            )

        # Mark the task as timed out (sets status to TIMEOUT)
        task.mark_timeout()

        # Emit task.timeout audit event
        self._emit_timeout_audit_event(task, elapsed_seconds)

        # Invoke callback if configured
        if self.on_timeout:
            try:
                self.on_timeout(task)
            except Exception as e:
                logger.exception(
                    "Error in on_timeout callback for task %s: %s", task.research_id, e
                )

    def _emit_timeout_audit_event(
        self, task: "BackgroundTask", elapsed_seconds: float
    ) -> None:
        """Emit a task.timeout audit event.

        Args:
            task: The BackgroundTask that timed out.
            elapsed_seconds: How long the task ran before timing out.
        """
        try:
            from foundry_mcp.core.observability import audit_log

            audit_log(
                "task_timeout",
                task_id=task.research_id,
                elapsed_seconds=round(elapsed_seconds, 2),
                configured_timeout=task.timeout,
                timed_out_at=task.timed_out_at,
            )
        except Exception as e:
            logger.debug("Failed to emit timeout audit event: %s", e)

    async def _handle_stale(self, task: "BackgroundTask") -> None:
        """Handle a stale task.

        Logs staleness detection, emits audit event, and invokes the on_stale
        callback if configured. Does not automatically cancel stale tasks.

        Args:
            task: The BackgroundTask that has become stale.
        """
        import time

        inactive_seconds = time.time() - task.last_activity
        logger.warning(
            "Task %s is stale (no activity for %.1fs, threshold=%.1fs)",
            task.research_id,
            inactive_seconds,
            self.stale_threshold,
        )

        # Emit task.stale audit event
        self._emit_stale_audit_event(task, inactive_seconds)

        # Invoke callback if configured
        if self.on_stale:
            try:
                self.on_stale(task)
            except Exception as e:
                logger.exception(
                    "Error in on_stale callback for task %s: %s", task.research_id, e
                )

    def _emit_stale_audit_event(
        self, task: "BackgroundTask", inactive_seconds: float
    ) -> None:
        """Emit a task.stale audit event.

        Args:
            task: The BackgroundTask that became stale.
            inactive_seconds: How long the task has been inactive.
        """
        try:
            from foundry_mcp.core.observability import audit_log

            audit_log(
                "task_stale",
                task_id=task.research_id,
                inactive_seconds=round(inactive_seconds, 2),
                stale_threshold=self.stale_threshold,
                last_activity=task.last_activity,
                elapsed_seconds=round(task.elapsed_ms / 1000, 2),
            )
        except Exception as e:
            logger.debug("Failed to emit stale audit event: %s", e)


# Module-level singleton for application-wide watchdog
_watchdog: Optional[TimeoutWatchdog] = None


def get_watchdog() -> Optional[TimeoutWatchdog]:
    """Get the global timeout watchdog instance.

    Returns:
        The global TimeoutWatchdog instance, or None if not initialized.
    """
    return _watchdog


def set_watchdog(watchdog: Optional[TimeoutWatchdog]) -> None:
    """Set the global timeout watchdog instance.

    Args:
        watchdog: TimeoutWatchdog instance to set as global, or None to clear.
    """
    global _watchdog
    _watchdog = watchdog


async def start_watchdog(
    poll_interval: float = 10.0,
    stale_threshold: float = 300.0,
    on_timeout: Optional[Callable[["BackgroundTask"], None]] = None,
    on_stale: Optional[Callable[["BackgroundTask"], None]] = None,
) -> TimeoutWatchdog:
    """Create and start the global timeout watchdog.

    Convenience function to create a TimeoutWatchdog, set it as the
    global instance, and start it.

    Args:
        poll_interval: Seconds between polling cycles (default 10).
        stale_threshold: Seconds without progress before task is stale (default 300).
        on_timeout: Optional callback when a task times out.
        on_stale: Optional callback when a task becomes stale.

    Returns:
        The started TimeoutWatchdog instance.
    """
    global _watchdog

    if _watchdog is not None and _watchdog.is_running:
        logger.warning("Global watchdog already running, stopping first")
        await _watchdog.stop()

    _watchdog = TimeoutWatchdog(
        poll_interval=poll_interval,
        stale_threshold=stale_threshold,
        on_timeout=on_timeout,
        on_stale=on_stale,
    )
    await _watchdog.start()
    return _watchdog


async def stop_watchdog(timeout: float = 5.0) -> None:
    """Stop the global timeout watchdog.

    Convenience function to stop the global watchdog instance.

    Args:
        timeout: Maximum seconds to wait for graceful shutdown (default 5).
    """
    global _watchdog

    if _watchdog is not None:
        await _watchdog.stop(timeout=timeout)
        _watchdog = None
