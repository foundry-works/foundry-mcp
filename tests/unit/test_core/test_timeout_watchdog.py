"""Tests for TimeoutWatchdog polling and monitoring.

Verifies:
- Polling interval respects configuration
- Timeout detection triggers callbacks
- Staleness detection triggers callbacks
- Watchdog lifecycle (start/stop)
- Concurrent timeout handling across multiple tasks
- Poll loop resilience to exceptions
- Callback exception isolation
- Module-level singleton management
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


class TestTimeoutWatchdogConcurrentTimeouts:
    """Tests for multiple tasks timing out in the same poll cycle."""

    @pytest.mark.asyncio
    async def test_multiple_tasks_timeout_same_cycle(self):
        """All timed-out tasks are handled when multiple timeout simultaneously."""
        timeout_ids = []

        def on_timeout(task):
            timeout_ids.append(task.research_id)

        watchdog = TimeoutWatchdog(poll_interval=0.01, on_timeout=on_timeout)

        from foundry_mcp.core.background_task import TaskStatus

        tasks = {}
        for i in range(5):
            t = MagicMock()
            t.research_id = f"concurrent-{i}"
            t.status = TaskStatus.RUNNING
            t.is_timed_out = True
            t.is_stale = MagicMock(return_value=False)
            t.elapsed_ms = 10000
            t.timeout = 1.0
            t.timed_out_at = None
            t.force_cancel = MagicMock()
            t.mark_timeout = MagicMock()
            tasks[f"concurrent-{i}"] = t

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = tasks

            await watchdog._check_tasks()

        assert sorted(timeout_ids) == [f"concurrent-{i}" for i in range(5)]
        for t in tasks.values():
            t.force_cancel.assert_called_once()
            t.mark_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_timeout_and_stale_tasks(self):
        """Timed-out and stale tasks are each handled correctly in one cycle."""
        timeout_ids = []
        stale_ids = []

        def on_timeout(task):
            timeout_ids.append(task.research_id)

        def on_stale(task):
            stale_ids.append(task.research_id)

        watchdog = TimeoutWatchdog(
            poll_interval=0.01, on_timeout=on_timeout, on_stale=on_stale
        )

        from foundry_mcp.core.background_task import TaskStatus

        timed_out_task = MagicMock()
        timed_out_task.research_id = "timeout-1"
        timed_out_task.status = TaskStatus.RUNNING
        timed_out_task.is_timed_out = True
        timed_out_task.elapsed_ms = 10000
        timed_out_task.timeout = 1.0
        timed_out_task.timed_out_at = None
        timed_out_task.force_cancel = MagicMock()
        timed_out_task.mark_timeout = MagicMock()

        stale_task = MagicMock()
        stale_task.research_id = "stale-1"
        stale_task.status = TaskStatus.RUNNING
        stale_task.is_timed_out = False
        stale_task.is_stale = MagicMock(return_value=True)
        stale_task.last_activity = 0
        stale_task.elapsed_ms = 50000

        healthy_task = MagicMock()
        healthy_task.research_id = "healthy-1"
        healthy_task.status = TaskStatus.RUNNING
        healthy_task.is_timed_out = False
        healthy_task.is_stale = MagicMock(return_value=False)

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {
                "timeout-1": timed_out_task,
                "stale-1": stale_task,
                "healthy-1": healthy_task,
            }

            await watchdog._check_tasks()

        assert timeout_ids == ["timeout-1"]
        assert stale_ids == ["stale-1"]

    @pytest.mark.asyncio
    async def test_timeout_takes_priority_over_stale(self):
        """A task that is both timed out and stale is handled as timed out only."""
        timeout_ids = []
        stale_ids = []

        def on_timeout(task):
            timeout_ids.append(task.research_id)

        def on_stale(task):
            stale_ids.append(task.research_id)

        watchdog = TimeoutWatchdog(
            poll_interval=0.01, on_timeout=on_timeout, on_stale=on_stale
        )

        from foundry_mcp.core.background_task import TaskStatus

        # Task is both timed out and stale
        task = MagicMock()
        task.research_id = "both-1"
        task.status = TaskStatus.RUNNING
        task.is_timed_out = True
        task.is_stale = MagicMock(return_value=True)
        task.elapsed_ms = 10000
        task.timeout = 1.0
        task.timed_out_at = None
        task.force_cancel = MagicMock()
        task.mark_timeout = MagicMock()

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"both-1": task}

            await watchdog._check_tasks()

        # Timeout handler fires, but NOT stale handler (continue skips stale check)
        assert timeout_ids == ["both-1"]
        assert stale_ids == []


class TestTimeoutWatchdogPollLoopResilience:
    """Tests for poll loop resilience to exceptions."""

    @pytest.mark.asyncio
    async def test_check_tasks_exception_does_not_crash_loop(self):
        """Poll loop continues after _check_tasks raises an exception."""
        call_count = 0

        async def failing_check():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("simulated failure")
            # Third call succeeds and stops the loop
            watchdog._stop_event.set()

        watchdog = TimeoutWatchdog(poll_interval=0.01)
        watchdog._check_tasks = failing_check

        await watchdog.start()
        await asyncio.sleep(0.1)
        await watchdog.stop()

        assert call_count >= 3, f"Expected at least 3 calls despite errors, got {call_count}"

    @pytest.mark.asyncio
    async def test_force_cancel_exception_does_not_crash_check(self):
        """Exception in force_cancel during timeout handling is caught."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        from foundry_mcp.core.background_task import TaskStatus

        task = MagicMock()
        task.research_id = "cancel-error-1"
        task.status = TaskStatus.RUNNING
        task.is_timed_out = True
        task.elapsed_ms = 5000
        task.timeout = 1.0
        task.timed_out_at = None
        task.force_cancel = MagicMock(side_effect=RuntimeError("cancel failed"))
        task.mark_timeout = MagicMock()

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"cancel-error-1": task}

            # Should not raise despite force_cancel error
            await watchdog._check_tasks()

        # mark_timeout still called even if force_cancel failed
        task.mark_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_timeout_callback_exception_is_isolated(self):
        """Exception in on_timeout callback doesn't crash watchdog or prevent other tasks."""

        def bad_callback(task):
            raise ValueError("callback boom")

        watchdog = TimeoutWatchdog(poll_interval=0.01, on_timeout=bad_callback)

        from foundry_mcp.core.background_task import TaskStatus

        task1 = MagicMock()
        task1.research_id = "cb-err-1"
        task1.status = TaskStatus.RUNNING
        task1.is_timed_out = True
        task1.elapsed_ms = 5000
        task1.timeout = 1.0
        task1.timed_out_at = None
        task1.force_cancel = MagicMock()
        task1.mark_timeout = MagicMock()

        task2 = MagicMock()
        task2.research_id = "cb-err-2"
        task2.status = TaskStatus.RUNNING
        task2.is_timed_out = True
        task2.elapsed_ms = 8000
        task2.timeout = 2.0
        task2.timed_out_at = None
        task2.force_cancel = MagicMock()
        task2.mark_timeout = MagicMock()

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"cb-err-1": task1, "cb-err-2": task2}

            # Should not raise despite callback error
            await watchdog._check_tasks()

        # Both tasks should still have been processed
        task1.force_cancel.assert_called_once()
        task1.mark_timeout.assert_called_once()
        task2.force_cancel.assert_called_once()
        task2.mark_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_stale_callback_exception_is_isolated(self):
        """Exception in on_stale callback doesn't crash the watchdog."""

        def bad_stale_callback(task):
            raise ValueError("stale callback boom")

        watchdog = TimeoutWatchdog(
            poll_interval=0.01, on_stale=bad_stale_callback
        )

        from foundry_mcp.core.background_task import TaskStatus

        task = MagicMock()
        task.research_id = "stale-cb-err"
        task.status = TaskStatus.RUNNING
        task.is_timed_out = False
        task.is_stale = MagicMock(return_value=True)
        task.last_activity = 0
        task.elapsed_ms = 50000

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {"stale-cb-err": task}

            # Should not raise
            await watchdog._check_tasks()

    @pytest.mark.asyncio
    async def test_empty_registry_no_errors(self):
        """Check tasks with empty registry completes without error."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = {}

            await watchdog._check_tasks()  # Should not raise


class TestTimeoutWatchdogRestartLifecycle:
    """Tests for watchdog restart and re-entrant behavior."""

    @pytest.mark.asyncio
    async def test_start_after_stop_creates_new_task(self):
        """Watchdog can be restarted after being stopped."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        await watchdog.start()
        task1 = watchdog._task
        assert watchdog.is_running

        await watchdog.stop()
        assert not watchdog.is_running

        await watchdog.start()
        task2 = watchdog._task
        assert watchdog.is_running
        assert task1 is not task2, "Should create a new asyncio task"

        await watchdog.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self):
        """Calling stop() on a never-started watchdog is safe."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)
        await watchdog.stop()  # Should not raise
        assert not watchdog.is_running

    @pytest.mark.asyncio
    async def test_multiple_restart_cycles(self):
        """Watchdog can be started and stopped multiple times."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        for _ in range(3):
            await watchdog.start()
            assert watchdog.is_running
            await watchdog.stop()
            assert not watchdog.is_running


class TestTimeoutWatchdogTaskStatusFiltering:
    """Tests for filtering tasks across all terminal statuses."""

    @pytest.mark.asyncio
    async def test_skips_all_terminal_statuses(self):
        """Watchdog skips tasks in COMPLETED, FAILED, CANCELLED, TIMEOUT states."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        from foundry_mcp.core.background_task import TaskStatus

        terminal_statuses = [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
            TaskStatus.PENDING,
        ]

        tasks = {}
        for i, status in enumerate(terminal_statuses):
            t = MagicMock()
            t.research_id = f"terminal-{i}"
            t.status = status
            # These would trigger if status filter didn't work
            t.is_timed_out = True
            t.is_stale = MagicMock(return_value=True)
            tasks[f"terminal-{i}"] = t

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry:
            mock_registry.return_value = tasks

            await watchdog._check_tasks()

        # None of the terminal tasks should have is_stale checked
        for t in tasks.values():
            t.is_stale.assert_not_called()


class TestTimeoutWatchdogModuleSingletons:
    """Tests for module-level singleton management functions."""

    @pytest.mark.asyncio
    async def test_get_watchdog_returns_none_initially(self):
        """get_watchdog returns None when no watchdog has been set."""
        from foundry_mcp.core.timeout_watchdog import get_watchdog, set_watchdog

        # Save and clear current state
        original = get_watchdog()
        set_watchdog(None)

        try:
            assert get_watchdog() is None
        finally:
            set_watchdog(original)

    @pytest.mark.asyncio
    async def test_set_and_get_watchdog(self):
        """set_watchdog stores instance retrievable by get_watchdog."""
        from foundry_mcp.core.timeout_watchdog import get_watchdog, set_watchdog

        original = get_watchdog()

        try:
            wd = TimeoutWatchdog(poll_interval=1.0)
            set_watchdog(wd)
            assert get_watchdog() is wd
        finally:
            set_watchdog(original)

    @pytest.mark.asyncio
    async def test_start_watchdog_creates_and_starts(self):
        """start_watchdog creates, sets, and starts a global watchdog."""
        from foundry_mcp.core.timeout_watchdog import (
            get_watchdog,
            set_watchdog,
            start_watchdog,
            stop_watchdog,
        )

        original = get_watchdog()
        set_watchdog(None)

        try:
            wd = await start_watchdog(poll_interval=0.01, stale_threshold=60.0)
            assert wd.is_running
            assert wd.poll_interval == 0.01
            assert wd.stale_threshold == 60.0
            assert get_watchdog() is wd

            await stop_watchdog()
            assert get_watchdog() is None
        finally:
            set_watchdog(original)

    @pytest.mark.asyncio
    async def test_start_watchdog_stops_existing_before_starting(self):
        """start_watchdog stops an already-running watchdog before creating a new one."""
        from foundry_mcp.core.timeout_watchdog import (
            get_watchdog,
            set_watchdog,
            start_watchdog,
            stop_watchdog,
        )

        original = get_watchdog()
        set_watchdog(None)

        try:
            wd1 = await start_watchdog(poll_interval=0.01)
            assert wd1.is_running

            wd2 = await start_watchdog(poll_interval=0.02)
            assert wd2.is_running
            assert not wd1.is_running
            assert wd2 is not wd1

            await stop_watchdog()
        finally:
            set_watchdog(original)

    @pytest.mark.asyncio
    async def test_stop_watchdog_when_none_is_noop(self):
        """stop_watchdog with no active watchdog is safe."""
        from foundry_mcp.core.timeout_watchdog import (
            get_watchdog,
            set_watchdog,
            stop_watchdog,
        )

        original = get_watchdog()
        set_watchdog(None)

        try:
            await stop_watchdog()  # Should not raise
            assert get_watchdog() is None
        finally:
            set_watchdog(original)

    @pytest.mark.asyncio
    async def test_start_watchdog_with_callbacks(self):
        """start_watchdog correctly passes on_timeout and on_stale callbacks."""
        from foundry_mcp.core.timeout_watchdog import (
            get_watchdog,
            set_watchdog,
            start_watchdog,
            stop_watchdog,
        )

        original = get_watchdog()
        set_watchdog(None)

        on_timeout_cb = MagicMock()
        on_stale_cb = MagicMock()

        try:
            wd = await start_watchdog(
                poll_interval=0.01,
                on_timeout=on_timeout_cb,
                on_stale=on_stale_cb,
            )
            assert wd.on_timeout is on_timeout_cb
            assert wd.on_stale is on_stale_cb

            await stop_watchdog()
        finally:
            set_watchdog(original)


class TestTimeoutWatchdogAuditEvents:
    """Tests for audit event emission during timeout/stale handling."""

    @pytest.mark.asyncio
    async def test_timeout_emits_audit_event(self):
        """Timeout handling emits task_timeout audit event."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        from foundry_mcp.core.background_task import TaskStatus

        task = MagicMock()
        task.research_id = "audit-timeout-1"
        task.status = TaskStatus.RUNNING
        task.is_timed_out = True
        task.elapsed_ms = 5000
        task.timeout = 1.0
        task.timed_out_at = None
        task.force_cancel = MagicMock()
        task.mark_timeout = MagicMock()

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry, patch(
            "foundry_mcp.core.timeout_watchdog.TimeoutWatchdog._emit_timeout_audit_event"
        ) as mock_audit:
            mock_registry.return_value = {"audit-timeout-1": task}

            await watchdog._check_tasks()

        mock_audit.assert_called_once()
        call_args = mock_audit.call_args
        assert call_args[0][0] is task
        assert isinstance(call_args[0][1], float)  # elapsed_seconds

    @pytest.mark.asyncio
    async def test_stale_emits_audit_event(self):
        """Stale handling emits task_stale audit event."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        from foundry_mcp.core.background_task import TaskStatus

        task = MagicMock()
        task.research_id = "audit-stale-1"
        task.status = TaskStatus.RUNNING
        task.is_timed_out = False
        task.is_stale = MagicMock(return_value=True)
        task.last_activity = 0
        task.elapsed_ms = 50000

        with patch(
            "foundry_mcp.core.task_registry.get_task_registry_async"
        ) as mock_registry, patch(
            "foundry_mcp.core.timeout_watchdog.TimeoutWatchdog._emit_stale_audit_event"
        ) as mock_audit:
            mock_registry.return_value = {"audit-stale-1": task}

            await watchdog._check_tasks()

        mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_event_failure_does_not_propagate(self):
        """Failed audit_log call is caught and doesn't crash handling."""
        watchdog = TimeoutWatchdog(poll_interval=0.01)

        task = MagicMock()
        task.research_id = "audit-fail-1"
        task.elapsed_ms = 5000
        task.timeout = 1.0
        task.timed_out_at = None

        with patch(
            "foundry_mcp.core.observability.audit_log",
            side_effect=RuntimeError("audit broken"),
        ):
            # Should not raise
            watchdog._emit_timeout_audit_event(task, 5.0)
            watchdog._emit_stale_audit_event(task, 300.0)
