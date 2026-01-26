"""Tests for BackgroundTask lifecycle and cancellation state transitions.

Verifies:
- State transitions: RUNNING -> CANCELLED
- Two-phase cancellation: cooperative then forced
- Cancellation event signaling during mid-phase execution
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.background_task import BackgroundTask, TaskStatus


class TestBackgroundTaskStateTransitions:
    """Tests for BackgroundTask state transitions during cancellation."""

    def test_initial_state_is_running(self):
        """Task starts in RUNNING state."""
        task = BackgroundTask(research_id="test-1")
        assert task.status == TaskStatus.RUNNING

    def test_thread_cancel_transitions_to_cancelled(self):
        """Thread-based task transitions from RUNNING to CANCELLED on cancel()."""
        # Create a thread that runs for a while
        stop_event = threading.Event()

        def worker():
            while not stop_event.is_set():
                time.sleep(0.01)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        task = BackgroundTask(research_id="test-1", thread=thread)
        assert task.status == TaskStatus.RUNNING

        # Cancel with short timeout (worker will stop quickly via stop_event)
        stop_event.set()  # Signal worker to stop
        result = task.cancel(timeout=1.0)

        assert result is True
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None

    def test_cancel_sets_event_before_status_change(self):
        """Cancellation event is set before status changes (intermediate state)."""
        # Track when event is set vs when status changes
        event_set_time = None
        status_when_event_set = None

        def worker(task: BackgroundTask):
            nonlocal event_set_time, status_when_event_set
            # Wait for cancellation signal
            while not task._cancel_event.is_set():
                time.sleep(0.01)
            # Record state at moment of cancellation signal
            event_set_time = time.time()
            status_when_event_set = task.status

        task = BackgroundTask(research_id="test-1")
        thread = threading.Thread(target=worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Give worker time to start
        time.sleep(0.05)

        # Cancel - this sets event first, then waits, then sets status
        task.cancel(timeout=1.0)

        # Verify the intermediate state was RUNNING when event was set
        # (because status change happens after join/wait)
        assert status_when_event_set == TaskStatus.RUNNING
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_event_signals_cooperative_shutdown(self):
        """Worker can check is_cancelled during cooperative shutdown window."""
        shutdown_detected = threading.Event()

        def cooperative_worker(task: BackgroundTask):
            # Simulate work with cancellation checks
            for _ in range(100):
                if task.is_cancelled:
                    shutdown_detected.set()
                    return  # Cooperative shutdown
                time.sleep(0.01)

        task = BackgroundTask(research_id="test-1")
        thread = threading.Thread(target=cooperative_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Give worker time to start
        time.sleep(0.05)

        # Cancel - worker should detect and exit cooperatively
        task.cancel(timeout=2.0)

        assert shutdown_detected.is_set()
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_returns_false_if_already_done(self):
        """Cancel returns False if task already completed."""
        # Create a task with a thread that's already done
        done_event = threading.Event()
        done_event.set()

        def quick_worker():
            pass  # Exits immediately

        thread = threading.Thread(target=quick_worker, daemon=True)
        thread.start()
        thread.join()  # Wait for completion

        task = BackgroundTask(research_id="test-1", thread=thread)

        # Try to cancel already-completed task
        result = task.cancel(timeout=0.1)

        assert result is False

    def test_force_cancel_after_timeout(self):
        """Task is marked cancelled even if thread doesn't stop cooperatively."""
        # Create a worker that ignores cancellation
        def stubborn_worker(task: BackgroundTask):
            # Deliberately ignores cancellation event
            time.sleep(10)

        task = BackgroundTask(research_id="test-1")
        thread = threading.Thread(target=stubborn_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Cancel with very short timeout
        task.cancel(timeout=0.1)

        # Status should be CANCELLED even though thread is still running
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None
        # Thread may still be alive (daemon, will be killed on exit)


class TestBackgroundTaskAsyncCancellation:
    """Tests for asyncio-based task cancellation."""

    @pytest.mark.asyncio
    async def test_asyncio_cancel_transitions_to_cancelled(self):
        """Asyncio task transitions from RUNNING to CANCELLED on cancel()."""

        async def async_worker():
            await asyncio.sleep(10)

        asyncio_task = asyncio.create_task(async_worker())
        task = BackgroundTask(research_id="test-1", task=asyncio_task)

        assert task.status == TaskStatus.RUNNING

        # Cancel
        result = task.cancel(timeout=0.1)

        assert result is True
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None

        # Clean up
        try:
            await asyncio_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_async_cancel_event_set(self):
        """Async cancellation event is set for cooperative shutdown."""

        async def async_worker(task: BackgroundTask):
            while not task.is_async_cancelled:
                await asyncio.sleep(0.01)
            return "shutdown"

        task = BackgroundTask(research_id="test-1")
        asyncio_task = asyncio.create_task(async_worker(task))
        task.task = asyncio_task

        # Give worker time to start
        await asyncio.sleep(0.05)

        # Cancel - sets async event
        task.cancel(timeout=0.1)

        assert task.is_async_cancelled
        assert task.status == TaskStatus.CANCELLED


class TestBackgroundTaskMarkMethods:
    """Tests for mark_completed, mark_failed, mark_timeout methods."""

    def test_mark_completed_sets_status_and_timestamp(self):
        """mark_completed sets COMPLETED status and completed_at."""
        task = BackgroundTask(research_id="test-1")
        result = MagicMock()

        task.mark_completed(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result is result

    def test_mark_completed_with_error_sets_failed_status(self):
        """mark_completed with error sets FAILED status and error message."""
        task = BackgroundTask(research_id="test-1")

        task.mark_completed(error="Something went wrong")

        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None
        assert task.error == "Something went wrong"

    def test_mark_timeout_sets_status(self):
        """mark_timeout sets TIMEOUT status."""
        task = BackgroundTask(research_id="test-1")

        task.mark_timeout()

        assert task.status == TaskStatus.TIMEOUT
        assert task.completed_at is not None


class TestBackgroundTaskProperties:
    """Tests for BackgroundTask property methods."""

    def test_elapsed_ms_increases_while_running(self):
        """elapsed_ms increases while task is running."""
        task = BackgroundTask(research_id="test-1")

        time.sleep(0.05)
        elapsed1 = task.elapsed_ms

        time.sleep(0.05)
        elapsed2 = task.elapsed_ms

        assert elapsed2 > elapsed1
        assert elapsed1 >= 50  # At least 50ms

    def test_elapsed_ms_frozen_after_completion(self):
        """elapsed_ms is frozen after task completes."""
        task = BackgroundTask(research_id="test-1")

        time.sleep(0.05)
        task.mark_completed(None)

        elapsed1 = task.elapsed_ms
        time.sleep(0.05)
        elapsed2 = task.elapsed_ms

        assert elapsed1 == elapsed2  # Should be same (frozen)

    def test_is_timed_out_respects_timeout(self):
        """is_timed_out returns True when timeout exceeded."""
        task = BackgroundTask(research_id="test-1", timeout=0.05)

        assert not task.is_timed_out

        time.sleep(0.06)

        assert task.is_timed_out

    def test_is_timed_out_false_without_timeout(self):
        """is_timed_out is False when no timeout set."""
        task = BackgroundTask(research_id="test-1", timeout=None)

        time.sleep(0.05)

        assert not task.is_timed_out

    def test_is_done_for_thread(self):
        """is_done reflects thread completion state."""

        def quick_worker():
            time.sleep(0.01)

        thread = threading.Thread(target=quick_worker, daemon=True)
        thread.start()

        task = BackgroundTask(research_id="test-1", thread=thread)

        # Thread is running
        assert not task.is_done

        # Wait for thread to complete
        thread.join()

        assert task.is_done

    @pytest.mark.asyncio
    async def test_is_done_for_asyncio_task(self):
        """is_done reflects asyncio task completion state."""

        async def quick_worker():
            await asyncio.sleep(0.01)

        asyncio_task = asyncio.create_task(quick_worker())
        task = BackgroundTask(research_id="test-1", task=asyncio_task)

        # Task is running
        assert not task.is_done

        # Wait for task to complete
        await asyncio_task

        assert task.is_done

    def test_is_stale_respects_threshold(self):
        """is_stale returns True when task inactive beyond threshold."""
        task = BackgroundTask(research_id="test-1")

        # Task just started, not stale
        assert not task.is_stale(stale_threshold=0.05)

        # Wait beyond threshold
        time.sleep(0.06)

        # Should be stale now
        assert task.is_stale(stale_threshold=0.05)

    def test_touch_resets_staleness(self):
        """touch() resets last_activity, preventing staleness."""
        task = BackgroundTask(research_id="test-1")

        # Wait to become stale
        time.sleep(0.06)
        assert task.is_stale(stale_threshold=0.05)

        # Touch to reset activity
        task.touch()

        # Should no longer be stale
        assert not task.is_stale(stale_threshold=0.05)

    def test_is_stale_false_when_not_running(self):
        """is_stale returns False for non-RUNNING tasks."""
        task = BackgroundTask(research_id="test-1")

        # Wait to become stale
        time.sleep(0.06)
        assert task.is_stale(stale_threshold=0.05)

        # Complete the task
        task.mark_completed(None)

        # Should not be stale anymore (not RUNNING)
        assert not task.is_stale(stale_threshold=0.05)


class TestBackgroundTaskTimeoutMetadata:
    """Tests for timeout metadata persistence."""

    def test_mark_timeout_persists_metadata(self):
        """mark_timeout sets timed_out_at and timeout_elapsed_seconds."""
        task = BackgroundTask(research_id="test-1", timeout=0.05)

        # Wait for timeout
        time.sleep(0.06)

        # Mark as timed out
        task.mark_timeout()

        assert task.status == TaskStatus.TIMEOUT
        assert task.timed_out_at is not None
        assert task.timeout_elapsed_seconds is not None
        assert task.timeout_elapsed_seconds >= 0.05
        assert task.completed_at == task.timed_out_at

    def test_timeout_metadata_not_set_for_completed_task(self):
        """Timeout metadata remains None for normally completed tasks."""
        task = BackgroundTask(research_id="test-1", timeout=10.0)

        task.mark_completed(result="done")

        assert task.status == TaskStatus.COMPLETED
        assert task.timed_out_at is None
        assert task.timeout_elapsed_seconds is None
