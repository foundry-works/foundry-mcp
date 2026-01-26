"""Tests for TaskRegistry concurrency and thread safety.

Verifies:
- Task registry add/get/remove operations
- Thread-safe concurrent access
- Async-safe concurrent access
- Cleanup of stale tasks
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.task_registry import (
    cleanup_stale_tasks,
    cleanup_stale_tasks_async,
    get,
    get_async,
    get_task_registry,
    get_task_registry_async,
    register,
    register_async,
    remove,
    remove_async,
    reset_task_registry,
    reset_task_registry_async,
)
from foundry_mcp.core.background_task import BackgroundTask, TaskStatus


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before and after each test."""
    reset_task_registry()
    yield
    reset_task_registry()


class TestTaskRegistryBasicOperations:
    """Tests for basic registry operations."""

    def test_register_and_get(self):
        """Should register and retrieve a task."""
        task = BackgroundTask(research_id="test-1")

        register(task)
        retrieved = get("test-1")

        assert retrieved is task

    def test_get_nonexistent_returns_none(self):
        """Should return None for nonexistent task."""
        result = get("nonexistent-id")

        assert result is None

    def test_remove_returns_task(self):
        """Should remove and return a task."""
        task = BackgroundTask(research_id="test-1")
        register(task)

        removed = remove("test-1")

        assert removed is task
        assert get("test-1") is None

    def test_remove_nonexistent_returns_none(self):
        """Should return None when removing nonexistent task."""
        result = remove("nonexistent-id")

        assert result is None

    def test_reset_clears_all_tasks(self):
        """Should clear all tasks from registry."""
        task1 = BackgroundTask(research_id="test-1")
        task2 = BackgroundTask(research_id="test-2")
        register(task1)
        register(task2)

        reset_task_registry()

        assert get("test-1") is None
        assert get("test-2") is None

    def test_get_task_registry_returns_dict(self):
        """Should return the registry dictionary."""
        task = BackgroundTask(research_id="test-1")
        register(task)

        registry = get_task_registry()

        assert isinstance(registry, dict)
        assert "test-1" in registry


class TestTaskRegistryAsyncOperations:
    """Tests for async registry operations."""

    @pytest.mark.asyncio
    async def test_register_and_get_async(self):
        """Should register and retrieve a task asynchronously."""
        task = BackgroundTask(research_id="test-async-1")

        await register_async(task)
        retrieved = await get_async("test-async-1")

        assert retrieved is task

    @pytest.mark.asyncio
    async def test_get_async_nonexistent_returns_none(self):
        """Should return None for nonexistent task."""
        result = await get_async("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_remove_async_returns_task(self):
        """Should remove and return a task asynchronously."""
        task = BackgroundTask(research_id="test-async-1")
        await register_async(task)

        removed = await remove_async("test-async-1")

        assert removed is task
        assert await get_async("test-async-1") is None

    @pytest.mark.asyncio
    async def test_reset_async_clears_all_tasks(self):
        """Should clear all tasks from registry asynchronously."""
        task1 = BackgroundTask(research_id="test-async-1")
        task2 = BackgroundTask(research_id="test-async-2")
        await register_async(task1)
        await register_async(task2)

        await reset_task_registry_async()

        assert await get_async("test-async-1") is None
        assert await get_async("test-async-2") is None

    @pytest.mark.asyncio
    async def test_get_task_registry_async_returns_dict(self):
        """Should return the registry dictionary asynchronously."""
        task = BackgroundTask(research_id="test-async-1")
        await register_async(task)

        registry = await get_task_registry_async()

        assert isinstance(registry, dict)
        assert "test-async-1" in registry


class TestTaskRegistryThreadConcurrency:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_register(self):
        """Should handle concurrent registrations without data loss."""
        num_tasks = 100
        tasks = [BackgroundTask(research_id=f"concurrent-{i}") for i in range(num_tasks)]

        def register_task(task):
            register(task)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(register_task, tasks)

        # Verify all tasks were registered
        registry = get_task_registry()
        assert len(registry) == num_tasks
        for i in range(num_tasks):
            assert get(f"concurrent-{i}") is not None

    def test_concurrent_get(self):
        """Should handle concurrent reads without errors."""
        task = BackgroundTask(research_id="shared-task")
        register(task)

        results = []
        errors = []

        def get_task():
            try:
                result = get("shared-task")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_task) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        assert all(r is task for r in results)

    def test_concurrent_register_and_remove(self):
        """Should handle concurrent register and remove operations."""
        registered_ids = set()
        removed_ids = set()
        lock = threading.Lock()

        def register_tasks():
            for i in range(50):
                task_id = f"task-{i}"
                task = BackgroundTask(research_id=task_id)
                register(task)
                with lock:
                    registered_ids.add(task_id)
                time.sleep(0.001)  # Yield to other threads

        def remove_tasks():
            for i in range(50):
                task_id = f"task-{i}"
                result = remove(task_id)
                if result is not None:
                    with lock:
                        removed_ids.add(task_id)
                time.sleep(0.001)  # Yield to other threads

        t1 = threading.Thread(target=register_tasks)
        t2 = threading.Thread(target=remove_tasks)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All registrations should have succeeded
        assert len(registered_ids) == 50

        # Some removes may have succeeded (race with registration)
        # Final state: registered - removed = remaining
        registry = get_task_registry()
        remaining = len(registry)
        assert remaining + len(removed_ids) <= 50

    def test_concurrent_mixed_operations(self):
        """Should handle mixed concurrent operations correctly."""
        # Pre-register some tasks
        for i in range(20):
            task = BackgroundTask(research_id=f"preexist-{i}")
            register(task)

        operations_completed = []
        errors = []
        lock = threading.Lock()

        def worker(worker_id):
            try:
                for i in range(10):
                    # Register
                    task = BackgroundTask(research_id=f"worker-{worker_id}-{i}")
                    register(task)

                    # Get (may or may not exist)
                    get(f"preexist-{i % 20}")

                    # Remove (may or may not exist)
                    remove(f"worker-{(worker_id + 1) % 5}-{i}")

                with lock:
                    operations_completed.append(worker_id)
            except Exception as e:
                with lock:
                    errors.append((worker_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(operations_completed) == 5


class TestTaskRegistryAsyncConcurrency:
    """Tests for async-safe concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_async_register(self):
        """Should handle concurrent async registrations."""
        num_tasks = 50

        async def register_task(i):
            task = BackgroundTask(research_id=f"async-concurrent-{i}")
            await register_async(task)

        await asyncio.gather(*[register_task(i) for i in range(num_tasks)])

        # Verify all tasks were registered
        for i in range(num_tasks):
            assert await get_async(f"async-concurrent-{i}") is not None

    @pytest.mark.asyncio
    async def test_concurrent_async_get(self):
        """Should handle concurrent async reads."""
        task = BackgroundTask(research_id="async-shared-task")
        await register_async(task)

        async def get_task():
            return await get_async("async-shared-task")

        results = await asyncio.gather(*[get_task() for _ in range(50)])

        assert all(r is task for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_async_mixed_operations(self):
        """Should handle mixed concurrent async operations."""
        # Pre-register some tasks
        for i in range(10):
            task = BackgroundTask(research_id=f"async-preexist-{i}")
            await register_async(task)

        async def worker(worker_id):
            for i in range(5):
                # Register
                task = BackgroundTask(research_id=f"async-worker-{worker_id}-{i}")
                await register_async(task)

                # Get
                await get_async(f"async-preexist-{i % 10}")

                # Remove
                await remove_async(f"async-worker-{(worker_id + 1) % 3}-{i}")

        # Run workers concurrently
        await asyncio.gather(*[worker(i) for i in range(3)])

        # Should complete without errors


class TestTaskRegistryCleanup:
    """Tests for stale task cleanup."""

    def test_cleanup_stale_tasks_removes_old_completed(self):
        """Should remove old completed tasks."""
        # Create a completed task with old timestamp
        task = BackgroundTask(research_id="old-task")
        task.mark_completed(result="done")
        task.completed_at = time.time() - 400  # 400 seconds ago
        register(task)

        # Create a recent completed task
        recent_task = BackgroundTask(research_id="recent-task")
        recent_task.mark_completed(result="done")
        register(recent_task)

        # Cleanup with 300 second threshold
        removed = cleanup_stale_tasks(max_age_seconds=300)

        assert removed == 1
        assert get("old-task") is None
        assert get("recent-task") is not None

    def test_cleanup_stale_tasks_keeps_running_tasks(self):
        """Should not remove running tasks regardless of age."""
        # Create a running task (not completed)
        task = BackgroundTask(research_id="running-task")
        # started_at is old but task is still running
        task.started_at = time.time() - 1000
        register(task)

        removed = cleanup_stale_tasks(max_age_seconds=300)

        assert removed == 0
        assert get("running-task") is not None

    def test_cleanup_stale_tasks_handles_empty_registry(self):
        """Should handle empty registry without errors."""
        removed = cleanup_stale_tasks(max_age_seconds=300)

        assert removed == 0

    @pytest.mark.asyncio
    async def test_cleanup_stale_tasks_async(self):
        """Should remove old completed tasks asynchronously."""
        # Create an old completed task
        task = BackgroundTask(research_id="async-old-task")
        task.mark_completed(result="done")
        task.completed_at = time.time() - 400
        await register_async(task)

        removed = await cleanup_stale_tasks_async(max_age_seconds=300)

        assert removed == 1
        assert await get_async("async-old-task") is None

    def test_cleanup_preserves_failed_tasks_within_threshold(self):
        """Should preserve recent failed tasks."""
        task = BackgroundTask(research_id="failed-task")
        task.mark_completed(error="Something failed")
        register(task)

        removed = cleanup_stale_tasks(max_age_seconds=300)

        assert removed == 0
        assert get("failed-task") is not None

    def test_cleanup_removes_old_cancelled_tasks(self):
        """Should remove old cancelled tasks."""
        # Create a thread to cancel (needs a thread to cancel properly)
        def worker():
            time.sleep(0.01)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        task = BackgroundTask(research_id="cancelled-task", thread=thread)
        thread.join()  # Wait for thread to finish
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time() - 400
        register(task)

        removed = cleanup_stale_tasks(max_age_seconds=300)

        assert removed == 1
        assert get("cancelled-task") is None


class TestTaskRegistryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_register_overwrites_existing(self):
        """Should overwrite existing task with same ID."""
        task1 = BackgroundTask(research_id="same-id")
        task2 = BackgroundTask(research_id="same-id")

        register(task1)
        register(task2)

        retrieved = get("same-id")
        assert retrieved is task2

    def test_concurrent_overwrite(self):
        """Should handle concurrent overwrites safely."""
        results = []

        def overwrite_task(value):
            task = BackgroundTask(research_id="overwrite-target")
            task.result = value  # Tag to identify which task won
            register(task)
            results.append(value)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(overwrite_task, range(100))

        # All 100 overwrites should have happened
        assert len(results) == 100

        # Final task should be one of the registered ones
        final = get("overwrite-target")
        assert final is not None
        assert final.result in range(100)
