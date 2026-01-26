"""Tests for TimeoutWatchdog polling and monitoring.

Verifies:
- Polling interval respects configuration
- Timeout detection triggers callbacks
- Staleness detection triggers callbacks
- Watchdog lifecycle (start/stop)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.timeout_watchdog import TimeoutWatchdog


class TestTimeoutWatchdogPollingInterval:
    """Tests for watchdog polling interval behavior."""

    @pytest.mark.asyncio
    async def test_poll_interval_configuration(self):
        """Watchdog respects configured poll_interval."""
        watchdog = TimeoutWatchdog(poll_interval=5.0)
        assert watchdog.poll_interval == 5.0

    @pytest.mark.asyncio
    async def test_default_poll_interval(self):
        """Watchdog uses default poll_interval of 10 seconds."""
        watchdog = TimeoutWatchdog()
        assert watchdog.poll_interval == 10.0

    @pytest.mark.asyncio
    async def test_poll_loop_calls_check_tasks(self):
        """Poll loop calls _check_tasks on each iteration."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)  # Very short for testing

        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1
            if check_count >= 3:
                # Stop after 3 checks
                watchdog._stop_event.set()

        watchdog._check_tasks = mock_check

        await watchdog.start()
        # Wait for a few poll cycles
        await asyncio.sleep(0.05)
        await watchdog.stop()

        assert check_count >= 3, f"Expected at least 3 check calls, got {check_count}"

    @pytest.mark.asyncio
    async def test_poll_loop_respects_interval_timing(self):
        """Poll loop waits approximately poll_interval between checks."""
        poll_interval = 0.05  # 50ms
        watchdog = TimeoutWatchdog(poll_interval=poll_interval)

        check_times = []

        async def mock_check():
            import time
            check_times.append(time.time())
            if len(check_times) >= 3:
                watchdog._stop_event.set()

        watchdog._check_tasks = mock_check

        await watchdog.start()
        await asyncio.sleep(0.2)  # Wait long enough for checks
        await watchdog.stop()

        # Verify at least 2 checks occurred
        assert len(check_times) >= 2

        # Check interval between calls (should be approximately poll_interval)
        for i in range(1, len(check_times)):
            interval = check_times[i] - check_times[i - 1]
            # Allow 20ms tolerance for timing variations
            assert interval >= poll_interval - 0.02, f"Interval {interval} too short"


class TestTimeoutWatchdogLifecycle:
    """Tests for watchdog start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running_state(self):
        """start() sets the watchdog to running state."""
        watchdog = TimeoutWatchdog(poll_interval=1.0)

        assert not watchdog.is_running

        await watchdog.start()

        assert watchdog.is_running
        assert watchdog._task is not None

        await watchdog.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running_state(self):
        """stop() clears the watchdog running state."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        await watchdog.start()
        assert watchdog.is_running

        await watchdog.stop()

        assert not watchdog.is_running
        assert watchdog._task is None

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Multiple start() calls don't create multiple tasks."""
        watchdog = TimeoutWatchdog(poll_interval=1.0)

        await watchdog.start()
        task1 = watchdog._task

        await watchdog.start()  # Second start
        task2 = watchdog._task

        assert task1 is task2, "Should be same task"

        await watchdog.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        """Multiple stop() calls don't cause errors."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        await watchdog.start()
        await watchdog.stop()
        await watchdog.stop()  # Second stop should be no-op

        assert not watchdog.is_running

    @pytest.mark.asyncio
    async def test_stop_with_slow_task_forces_cancel(self):
        """stop() cancels task if it doesn't stop gracefully."""
        watchdog = TimeoutWatchdog(poll_interval=10.0)

        # Make _check_tasks take a long time
        async def slow_check():
            await asyncio.sleep(100)

        watchdog._check_tasks = slow_check

        await watchdog.start()

        # Stop with very short timeout
        await watchdog.stop(timeout=0.1)

        assert not watchdog.is_running


class TestTimeoutWatchdogTimeoutDetection:
    """Tests for timeout detection and handling."""

    @pytest.mark.asyncio
    async def test_timeout_callback_invoked(self):
        """on_timeout callback is invoked when task times out."""
        timeout_tasks = []

        def on_timeout(task):
            timeout_tasks.append(task)

        watchdog = TimeoutWatchdog(poll_interval=0.01, on_timeout=on_timeout)

        # Create a mock task that is timed out
        mock_task = MagicMock()
        mock_task.research_id = "test-timeout-1"
        mock_task.status = MagicMock()
        mock_task.status.name = "RUNNING"
        mock_task.is_timed_out = True
        mock_task.is_stale = MagicMock(return_value=False)
        mock_task.elapsed_ms = 5000
        mock_task.timeout = 1.0
        mock_task.timed_out_at = None
        mock_task.force_cancel = MagicMock()
        mock_task.mark_timeout = MagicMock()

        # Mock TaskStatus enum
        from foundry_mcp.core.background_task import TaskStatus
        mock_task.status = TaskStatus.RUNNING

        # Mock the registry to return our task (patch at task_registry module)
        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"test-timeout-1": mock_task}

            await watchdog._check_tasks()

        assert len(timeout_tasks) == 1
        assert timeout_tasks[0] is mock_task
        mock_task.force_cancel.assert_called_once()
        mock_task.mark_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_triggers_cancellation(self):
        """Timed-out task triggers force_cancel."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        mock_task = MagicMock()
        mock_task.research_id = "test-timeout-2"
        mock_task.is_timed_out = True
        mock_task.is_stale = MagicMock(return_value=False)
        mock_task.elapsed_ms = 5000
        mock_task.timeout = 1.0
        mock_task.timed_out_at = None
        mock_task.force_cancel = MagicMock()
        mock_task.mark_timeout = MagicMock()

        from foundry_mcp.core.background_task import TaskStatus
        mock_task.status = TaskStatus.RUNNING

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"test-timeout-2": mock_task}

            await watchdog._check_tasks()

        mock_task.force_cancel.assert_called_once()


class TestTimeoutWatchdogStalenessDetection:
    """Tests for staleness detection and handling."""

    @pytest.mark.asyncio
    async def test_stale_callback_invoked(self):
        """on_stale callback is invoked when task becomes stale."""
        stale_tasks = []

        def on_stale(task):
            stale_tasks.append(task)

        watchdog = TimeoutWatchdog(
            poll_interval=0.01, stale_threshold=0.05, on_stale=on_stale
        )

        mock_task = MagicMock()
        mock_task.research_id = "test-stale-1"
        mock_task.is_timed_out = False
        mock_task.is_stale = MagicMock(return_value=True)
        mock_task.last_activity = 0  # Long time ago
        mock_task.elapsed_ms = 10000

        from foundry_mcp.core.background_task import TaskStatus
        mock_task.status = TaskStatus.RUNNING

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"test-stale-1": mock_task}

            await watchdog._check_tasks()

        assert len(stale_tasks) == 1
        assert stale_tasks[0] is mock_task

    @pytest.mark.asyncio
    async def test_stale_threshold_configuration(self):
        """Watchdog respects configured stale_threshold."""
        watchdog = TimeoutWatchdog(stale_threshold=120.0)
        assert watchdog.stale_threshold == 120.0

    @pytest.mark.asyncio
    async def test_default_stale_threshold(self):
        """Watchdog uses default stale_threshold of 300 seconds."""
        watchdog = TimeoutWatchdog()
        assert watchdog.stale_threshold == 300.0


class TestTimeoutWatchdogTaskFiltering:
    """Tests for task filtering logic."""

    @pytest.mark.asyncio
    async def test_only_checks_running_tasks(self):
        """Watchdog only checks tasks with RUNNING status."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        from foundry_mcp.core.background_task import TaskStatus

        running_task = MagicMock()
        running_task.research_id = "running-1"
        running_task.status = TaskStatus.RUNNING
        running_task.is_timed_out = False
        running_task.is_stale = MagicMock(return_value=False)

        completed_task = MagicMock()
        completed_task.research_id = "completed-1"
        completed_task.status = TaskStatus.COMPLETED
        # These should not be checked
        completed_task.is_timed_out = True  # Would trigger if checked
        completed_task.is_stale = MagicMock(return_value=True)

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {
                "running-1": running_task,
                "completed-1": completed_task,
            }

            await watchdog._check_tasks()

        # Running task should have is_stale checked
        running_task.is_stale.assert_called()
        # Completed task should not have is_stale checked
        completed_task.is_stale.assert_not_called()
