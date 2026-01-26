"""Tests for cancellation timing behavior.

Verifies:
- Cooperative cancellation completes within 5s timeout
- Forced cancellation triggers after timeout
- Timing behavior with mock providers
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.background_task import BackgroundTask, TaskStatus


class TestCooperativeCancellationTiming:
    """Tests for cooperative cancellation completing within timeout."""

    def test_cooperative_cancellation_completes_within_timeout(self):
        """Should complete cooperative cancellation within 5s timeout."""
        cancel_detected = threading.Event()
        worker_stopped = threading.Event()

        def cooperative_worker(task: BackgroundTask):
            """Worker that checks cancellation and exits cooperatively."""
            # Simulate work with frequent cancellation checks
            for _ in range(100):
                if task.is_cancelled:
                    cancel_detected.set()
                    worker_stopped.set()
                    return  # Exit cooperatively
                time.sleep(0.05)  # Simulate work
            worker_stopped.set()

        # Create task with thread
        task = BackgroundTask(research_id="coop-test-1")
        thread = threading.Thread(target=cooperative_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Give worker time to start
        time.sleep(0.1)

        # Measure cancellation time
        start_time = time.time()
        result = task.cancel(timeout=5.0)  # 5 second timeout
        elapsed = time.time() - start_time

        # Verify cooperative cancellation occurred
        assert result is True
        assert task.status == TaskStatus.CANCELLED
        assert cancel_detected.is_set()
        assert worker_stopped.is_set()
        # Should complete well within 5 seconds (cooperative shutdown)
        assert elapsed < 5.0, f"Cooperative cancellation took {elapsed:.2f}s (expected < 5s)"

    def test_cooperative_cancellation_with_quick_response(self):
        """Should handle very fast cooperative shutdown."""
        def quick_worker(task: BackgroundTask):
            """Worker that checks cancellation immediately."""
            while not task.is_cancelled:
                time.sleep(0.01)
            # Exit immediately on cancel

        task = BackgroundTask(research_id="quick-coop-test")
        thread = threading.Thread(target=quick_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)  # Let worker start

        start_time = time.time()
        task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        # Very quick shutdown - well under 1 second
        assert elapsed < 1.0, f"Quick cooperative cancellation took {elapsed:.2f}s"

    def test_cooperative_cancellation_at_iteration_boundary(self):
        """Should cancel at iteration boundary as designed."""
        iterations_completed = []
        cancelled_at_boundary = threading.Event()

        def iteration_worker(task: BackgroundTask):
            """Worker that checks cancellation at iteration boundaries."""
            for i in range(10):
                # Iteration boundary check (as implemented in deep_research)
                if task.is_cancelled:
                    cancelled_at_boundary.set()
                    return

                # Simulate iteration work
                iterations_completed.append(i)
                time.sleep(0.1)

        task = BackgroundTask(research_id="boundary-test")
        thread = threading.Thread(target=iteration_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Let a few iterations complete
        time.sleep(0.35)

        # Cancel - should stop at next boundary
        start_time = time.time()
        task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        assert cancelled_at_boundary.is_set()
        assert len(iterations_completed) >= 2  # At least 2 iterations completed
        assert len(iterations_completed) < 10  # But not all
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_asyncio_cooperative_cancellation_timing(self):
        """Should handle asyncio cooperative cancellation within timeout."""
        cancel_detected = asyncio.Event()

        async def async_worker(task: BackgroundTask):
            """Async worker that checks cancellation."""
            for _ in range(100):
                if task.is_async_cancelled:
                    cancel_detected.set()
                    return
                await asyncio.sleep(0.05)

        task = BackgroundTask(research_id="async-coop-test")
        asyncio_task = asyncio.create_task(async_worker(task))
        task.task = asyncio_task

        await asyncio.sleep(0.1)  # Let worker start

        start_time = time.time()
        task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        assert task.is_async_cancelled
        assert elapsed < 5.0

        # Clean up
        try:
            await asyncio_task
        except asyncio.CancelledError:
            pass


class TestForcedCancellationTiming:
    """Tests for forced cancellation after timeout."""

    def test_forced_cancellation_after_timeout(self):
        """Should transition to CANCELLED even if thread ignores signal."""
        def stubborn_worker(task: BackgroundTask):
            """Worker that ignores cancellation signal."""
            # Intentionally ignores task.is_cancelled
            time.sleep(10)  # Long sleep

        task = BackgroundTask(research_id="forced-test")
        thread = threading.Thread(target=stubborn_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)  # Let worker start

        # Cancel with short timeout - worker won't cooperate
        start_time = time.time()
        task.cancel(timeout=0.5)  # Short timeout
        elapsed = time.time() - start_time

        # Status should be CANCELLED (even though thread may still be running)
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None
        # Should complete after timeout (plus some margin)
        assert 0.4 < elapsed < 1.0, f"Forced cancellation took {elapsed:.2f}s"
        # Note: thread may still be running (daemon, will be killed on process exit)

    def test_immediate_forced_cancellation(self):
        """Should force cancel immediately with timeout=0."""
        def slow_worker(task: BackgroundTask):
            """Worker that would take long to stop."""
            for _ in range(100):
                if task.is_cancelled:
                    time.sleep(0.5)  # Slow cleanup
                    return
                time.sleep(0.1)

        task = BackgroundTask(research_id="immediate-force-test")
        thread = threading.Thread(target=slow_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)

        # Immediate force cancel
        start_time = time.time()
        task.cancel(timeout=0)  # No waiting
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        # Should be nearly instant (no waiting for cooperative shutdown)
        assert elapsed < 0.1, f"Immediate cancel took {elapsed:.2f}s"


class TestCancellationTimingWithMockProvider:
    """Integration tests with mock provider for realistic scenarios."""

    def test_cancel_during_provider_call(self):
        """Should cancel during a simulated provider call."""
        provider_call_started = threading.Event()
        provider_call_cancelled = threading.Event()

        def mock_provider_call(task: BackgroundTask):
            """Simulates a provider call that checks cancellation."""
            provider_call_started.set()
            # Simulate provider call with cancellation check
            for _ in range(50):
                if task.is_cancelled:
                    provider_call_cancelled.set()
                    return None  # Return partial/no result
                time.sleep(0.05)
            return "provider_result"

        def workflow_worker(task: BackgroundTask):
            """Simulates workflow calling provider."""
            result = mock_provider_call(task)
            if result is None:
                return  # Cancelled

        task = BackgroundTask(research_id="provider-cancel-test")
        thread = threading.Thread(target=workflow_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Wait for provider call to start
        provider_call_started.wait(timeout=1.0)
        time.sleep(0.1)

        # Cancel during provider call
        start_time = time.time()
        task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        assert provider_call_cancelled.is_set()
        assert elapsed < 5.0

    def test_cancel_between_provider_calls(self):
        """Should cancel between multiple provider calls."""
        calls_made = []

        def mock_provider(task: BackgroundTask, call_num: int):
            """Mock provider that takes some time."""
            time.sleep(0.1)
            return f"result_{call_num}"

        def multi_call_workflow(task: BackgroundTask):
            """Workflow that makes multiple provider calls."""
            for i in range(10):
                # Cancellation check between calls (as implemented)
                if task.is_cancelled:
                    return
                result = mock_provider(task, i)
                calls_made.append(result)

        task = BackgroundTask(research_id="multi-provider-test")
        thread = threading.Thread(target=multi_call_workflow, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        # Let a few calls complete
        time.sleep(0.35)

        # Cancel
        start_time = time.time()
        task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        assert len(calls_made) >= 2  # Some calls completed
        assert len(calls_made) < 10  # But not all
        assert elapsed < 5.0


class TestForcedCancellationWithBlockingProvider:
    """Integration tests for forced cancellation with blocking providers."""

    def test_forced_cancellation_with_blocking_provider_under_10s(self):
        """Should complete forced cancellation within 10s even with blocking provider."""
        provider_started = threading.Event()

        def blocking_provider():
            """Simulates a provider that blocks and ignores cancellation."""
            provider_started.set()
            # Simulate a blocking I/O call that can't be interrupted
            time.sleep(30)  # Would take 30s if not cancelled
            return "result"

        def workflow_with_blocking_provider(task: BackgroundTask):
            """Workflow using blocking provider."""
            # This provider doesn't check cancellation
            return blocking_provider()

        task = BackgroundTask(research_id="blocking-provider-test")
        thread = threading.Thread(
            target=workflow_with_blocking_provider, args=(task,), daemon=True
        )
        thread.start()
        task.thread = thread

        # Wait for provider to start
        provider_started.wait(timeout=1.0)

        # Cancel with 5s cooperative timeout - provider won't cooperate
        # Total time should still be under 10s (5s timeout + overhead)
        start_time = time.time()
        task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None
        # Should complete in ~5s (cooperative timeout) + small overhead
        assert elapsed < 10.0, f"Forced cancellation took {elapsed:.2f}s (expected < 10s)"

    def test_forced_cancellation_marks_status_immediately_after_timeout(self):
        """Should mark task as CANCELLED immediately after timeout expires."""
        def infinite_worker(task: BackgroundTask):
            """Worker that never checks cancellation."""
            while True:
                time.sleep(0.1)

        task = BackgroundTask(research_id="infinite-worker-test")
        thread = threading.Thread(target=infinite_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)

        # Use short timeout for faster test
        start_time = time.time()
        task.cancel(timeout=0.5)
        elapsed = time.time() - start_time

        # Status should be CANCELLED even though thread is still running
        assert task.status == TaskStatus.CANCELLED
        # Should complete at approximately the timeout value
        assert 0.4 < elapsed < 1.0, f"Took {elapsed:.2f}s (expected ~0.5s)"

    def test_forced_cancellation_with_slow_cleanup(self):
        """Should handle workers with slow cleanup after cancellation signal."""
        cleanup_started = threading.Event()
        cleanup_completed = threading.Event()

        def slow_cleanup_worker(task: BackgroundTask):
            """Worker with slow cleanup routine."""
            while not task.is_cancelled:
                time.sleep(0.01)
            # Slow cleanup that exceeds cooperative timeout
            cleanup_started.set()
            time.sleep(10)  # Very slow cleanup
            cleanup_completed.set()

        task = BackgroundTask(research_id="slow-cleanup-test")
        thread = threading.Thread(target=slow_cleanup_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)

        # Cancel with short timeout - cleanup won't finish in time
        start_time = time.time()
        task.cancel(timeout=0.5)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        assert cleanup_started.is_set()  # Cleanup was attempted
        assert not cleanup_completed.is_set()  # But didn't finish
        # Should return after timeout even though cleanup is ongoing
        assert elapsed < 1.0

    def test_forced_cancellation_total_time_guarantee(self):
        """Should guarantee total cancellation time is bounded."""
        def stubborn_worker(task: BackgroundTask):
            """Worker that completely ignores cancellation."""
            time.sleep(60)  # Would run for 60s

        task = BackgroundTask(research_id="stubborn-test")
        thread = threading.Thread(target=stubborn_worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)

        # Even with a stubborn worker, cancellation should complete
        # within timeout + small overhead
        start_time = time.time()
        task.cancel(timeout=2.0)
        elapsed = time.time() - start_time

        assert task.status == TaskStatus.CANCELLED
        # Total time bounded by timeout
        assert elapsed < 3.0, f"Total cancellation time {elapsed:.2f}s exceeded bound"


class TestCancellationTimingEdgeCases:
    """Edge cases for cancellation timing."""

    def test_cancel_already_completed_task(self):
        """Should return False instantly for completed task."""
        def quick_worker():
            pass  # Exits immediately

        thread = threading.Thread(target=quick_worker, daemon=True)
        thread.start()
        thread.join()  # Wait for completion

        task = BackgroundTask(research_id="completed-test", thread=thread)

        start_time = time.time()
        result = task.cancel(timeout=5.0)
        elapsed = time.time() - start_time

        assert result is False  # Already done
        assert elapsed < 0.1  # Nearly instant

    def test_cancel_without_thread_or_task(self):
        """Should handle cancellation of task without thread/asyncio task."""
        task = BackgroundTask(research_id="no-executor-test")

        result = task.cancel(timeout=5.0)

        assert result is False  # Nothing to cancel

    def test_multiple_cancel_calls(self):
        """Should handle multiple cancel calls gracefully."""
        cancel_count = []

        def worker(task: BackgroundTask):
            while not task.is_cancelled:
                time.sleep(0.01)
            cancel_count.append(1)

        task = BackgroundTask(research_id="multi-cancel-test")
        thread = threading.Thread(target=worker, args=(task,), daemon=True)
        thread.start()
        task.thread = thread

        time.sleep(0.05)

        # First cancel
        result1 = task.cancel(timeout=1.0)
        # Second cancel (task already cancelled)
        result2 = task.cancel(timeout=1.0)

        assert result1 is True
        assert result2 is False  # Already done
        assert task.status == TaskStatus.CANCELLED
