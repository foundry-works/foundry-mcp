"""Tests for the provider executor.

Tests cover:
- Load testing with concurrent blocking operations
- Graceful shutdown behavior
- Feature flag disabling executor isolation
- Fallback executor behavior
- Metrics recording
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.executor import (
    ProviderExecutor,
    ExecutorExhaustedError,
    configure_executor,
    get_provider_executor,
    DEFAULT_POOL_SIZE,
    DEFAULT_QUEUE_LIMIT,
)


class TestProviderExecutorLoadTest:
    """Load tests for executor with concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_blocking_operations_complete(self):
        """10 concurrent sessions with 500ms blocking ops complete within 2x expected time."""
        executor = ProviderExecutor(pool_size=4, queue_limit=100, enabled=True)
        executor.start()

        def blocking_operation() -> str:
            """Simulate a 100ms blocking operation."""
            time.sleep(0.1)
            return "done"

        try:
            start = time.time()

            # Run 10 concurrent operations
            tasks = [
                executor.run_blocking(blocking_operation)
                for _ in range(10)
            ]
            results = await asyncio.gather(*tasks)

            elapsed = time.time() - start

            # All should complete successfully
            assert len(results) == 10
            assert all(r == "done" for r in results)

            # With 4 workers, 10 x 100ms operations should complete in ~300ms
            # (3 batches: 4+4+2). Allow 2x margin = 600ms
            assert elapsed < 0.6, f"Expected < 600ms, got {elapsed*1000:.0f}ms"

        finally:
            await executor.shutdown()

    @pytest.mark.asyncio
    async def test_no_executor_exhausted_error_under_load(self):
        """Under normal load, no ExecutorExhaustedError should be raised."""
        executor = ProviderExecutor(pool_size=4, queue_limit=100, enabled=True)
        executor.start()

        def blocking_operation() -> str:
            time.sleep(0.05)
            return "done"

        try:
            # Run 20 concurrent operations (well within queue limit)
            tasks = [
                executor.run_blocking(blocking_operation)
                for _ in range(20)
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 20
        finally:
            await executor.shutdown()

    @pytest.mark.asyncio
    async def test_metrics_show_utilization(self):
        """Metrics should show active workers and queued tasks."""
        executor = ProviderExecutor(pool_size=2, queue_limit=100, enabled=True)
        executor.start()

        # Track metrics calls
        metrics_calls = []

        def mock_gauge(name: str, value: int) -> None:
            metrics_calls.append((name, value))

        with patch("foundry_mcp.core.executor.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_metrics.gauge = mock_gauge
            mock_metrics.counter = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            def blocking_operation() -> str:
                time.sleep(0.1)
                return "done"

            try:
                # Run operations to generate metrics
                tasks = [
                    executor.run_blocking(blocking_operation)
                    for _ in range(4)
                ]
                await asyncio.gather(*tasks)

                # Should have recorded active_workers and queued_tasks metrics
                metric_names = [name for name, _ in metrics_calls]
                assert "provider_executor_active_workers" in metric_names
                assert "provider_executor_queued_tasks" in metric_names

            finally:
                await executor.shutdown()


class TestProviderExecutorGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_pending_tasks(self):
        """Shutdown with wait=True should wait for pending tasks."""
        executor = ProviderExecutor(pool_size=2, enabled=True)
        executor.start()

        completed = []

        def slow_operation(idx: int) -> str:
            time.sleep(0.1)
            completed.append(idx)
            return f"done-{idx}"

        # Start some operations
        task1 = asyncio.create_task(executor.run_blocking(slow_operation, 1))
        task2 = asyncio.create_task(executor.run_blocking(slow_operation, 2))

        # Small delay to ensure tasks are submitted
        await asyncio.sleep(0.01)

        # Shutdown should wait for tasks
        await executor.shutdown(wait=True)

        # Wait for tasks to complete
        await asyncio.gather(task1, task2, return_exceptions=True)

        # Both should have completed
        assert 1 in completed
        assert 2 in completed

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Multiple shutdown calls should be safe."""
        executor = ProviderExecutor(pool_size=2, enabled=True)
        executor.start()

        # Multiple shutdown calls should not raise
        await executor.shutdown()
        await executor.shutdown()
        await executor.shutdown()

        assert not executor.is_started


class TestProviderExecutorFeatureFlag:
    """Tests for feature flag behavior."""

    @pytest.mark.asyncio
    async def test_disabled_executor_runs_inline(self):
        """When disabled, operations run inline without thread pool."""
        executor = ProviderExecutor(pool_size=4, enabled=False)
        executor.start()

        call_count = 0

        def inline_operation() -> str:
            nonlocal call_count
            call_count += 1
            return "inline"

        try:
            result = await executor.run_blocking(inline_operation)
            assert result == "inline"
            assert call_count == 1

            # Executor should not have created thread pool
            assert executor._executor is None
        finally:
            await executor.shutdown()

    @pytest.mark.asyncio
    async def test_disabled_context_manager(self):
        """disabled() context manager temporarily disables isolation."""
        executor = ProviderExecutor(pool_size=2, enabled=True)
        executor.start()

        try:
            assert executor.is_enabled

            with executor.disabled():
                assert not executor.is_enabled
                # Operations run inline within context
                result = await executor.run_blocking(lambda: "inline")
                assert result == "inline"

            # Restored after context
            assert executor.is_enabled
        finally:
            await executor.shutdown()


class TestProviderExecutorFallback:
    """Tests for fallback executor behavior."""

    @pytest.mark.asyncio
    async def test_fallback_used_when_queue_full(self):
        """Fallback executor used when primary queue is exhausted."""
        # Very small queue to trigger fallback
        executor = ProviderExecutor(pool_size=1, queue_limit=1, enabled=True)
        executor.start()

        def blocking_operation() -> str:
            time.sleep(0.1)
            return "done"

        try:
            # Submit multiple operations to overflow queue
            tasks = [
                executor.run_blocking(blocking_operation)
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)

            # All should complete (some via fallback)
            assert len(results) == 5
            assert all(r == "done" for r in results)

            # Fallback should have been used
            assert executor.fallback_count > 0

        finally:
            await executor.shutdown()


class TestProviderExecutorConfiguration:
    """Tests for executor configuration."""

    def test_configure_executor_returns_singleton(self):
        """configure_executor should configure the global singleton."""
        executor = configure_executor(
            pool_size=8,
            queue_limit=200,
            enabled=True,
        )

        assert executor._pool_size == 8
        assert executor._queue_limit == 200
        assert executor.is_enabled

    def test_get_provider_executor_returns_singleton(self):
        """get_provider_executor should return same instance."""
        exec1 = get_provider_executor()
        exec2 = get_provider_executor()
        assert exec1 is exec2

    def test_default_configuration(self):
        """Default configuration should use expected values."""
        executor = ProviderExecutor()
        assert executor._pool_size == DEFAULT_POOL_SIZE
        assert executor._queue_limit == DEFAULT_QUEUE_LIMIT
        assert executor.is_enabled
