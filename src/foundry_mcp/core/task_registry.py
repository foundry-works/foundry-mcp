"""Task registry for tracking background research tasks.

Provides a global singleton registry for storing and retrieving
background research tasks with thread-safe access.
"""

import threading
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from foundry_mcp.core.background_task import BackgroundTask


# Global registry instance
_registry: Dict[str, "BackgroundTask"] = {}
_registry_lock = threading.Lock()
def get_task_registry() -> Dict[str, "BackgroundTask"]:
    """Get the global task registry.

    Returns a dictionary mapping task IDs to BackgroundTask instances.
    The registry is thread-safe and maintains in-memory tracking of
    all active background research tasks.

    Returns:
        Dictionary of task_id -> BackgroundTask
    """
    global _registry
    with _registry_lock:
        return dict(_registry)


def reset_task_registry() -> None:
    """Reset the global task registry (for testing).

    Clears all registered tasks. This should only be used in test
    environments or during controlled shutdown.
    """
    global _registry
    with _registry_lock:
        _registry.clear()


async def get_task_registry_async() -> Dict[str, "BackgroundTask"]:
    """Get the global task registry (async version).

    Async-safe version of get_task_registry() for use in async contexts.
    Returns a dictionary mapping task IDs to BackgroundTask instances.
    The registry is async-locked and maintains in-memory tracking of
    all active background research tasks.

    Returns:
        Dictionary of task_id -> BackgroundTask
    """
    import asyncio

    return await asyncio.to_thread(get_task_registry)


async def reset_task_registry_async() -> None:
    """Reset the global task registry (async version for testing).

    Async-safe version of reset_task_registry() for use in async contexts.
    Clears all registered tasks. This should only be used in test
    environments or during controlled shutdown.
    """
    import asyncio

    await asyncio.to_thread(reset_task_registry)


def register(task: "BackgroundTask") -> None:
    """Register a background task in the global registry.

    Stores the task in the registry using its research_id as the key.
    The operation is thread-safe and uses the global registry lock.

    Args:
        task: BackgroundTask instance to register. Must have research_id attribute.
    """
    global _registry
    with _registry_lock:
        _registry[task.research_id] = task


async def register_async(task: "BackgroundTask") -> None:
    """Register a background task in the global registry (async version).

    Async-safe version of register() for use in async contexts.
    Stores the task in the registry using its research_id as the key.

    Args:
        task: BackgroundTask instance to register. Must have research_id attribute.
    """
    import asyncio

    await asyncio.to_thread(register, task)


def get(task_id: str) -> "BackgroundTask | None":
    """Retrieve a background task by ID from the registry.

    Looks up the task in the registry by its ID. Returns None if the task
    is not found. The operation is thread-safe.

    Args:
        task_id: The research_id of the task to retrieve.

    Returns:
        BackgroundTask instance if found, None otherwise.
    """
    global _registry
    with _registry_lock:
        return _registry.get(task_id)


async def get_async(task_id: str) -> "BackgroundTask | None":
    """Retrieve a background task by ID from the registry (async version).

    Async-safe version of get() for use in async contexts.
    Looks up the task in the registry by its ID. Returns None if the task
    is not found.

    Args:
        task_id: The research_id of the task to retrieve.

    Returns:
        BackgroundTask instance if found, None otherwise.
    """
    import asyncio

    return await asyncio.to_thread(get, task_id)


def remove(task_id: str) -> "BackgroundTask | None":
    """Remove and return a background task from the registry.

    Removes the task from the registry and returns it. If the task is not
    found, returns None. The operation is thread-safe.

    Args:
        task_id: The research_id of the task to remove.

    Returns:
        BackgroundTask instance if found and removed, None otherwise.
    """
    global _registry
    with _registry_lock:
        return _registry.pop(task_id, None)


async def remove_async(task_id: str) -> "BackgroundTask | None":
    """Remove and return a background task from the registry (async version).

    Async-safe version of remove() for use in async contexts.
    Removes the task from the registry and returns it. If the task is not
    found, returns None.

    Args:
        task_id: The research_id of the task to remove.

    Returns:
        BackgroundTask instance if found and removed, None otherwise.
    """
    import asyncio

    return await asyncio.to_thread(remove, task_id)


def cleanup_stale_tasks(max_age_seconds: float = 300) -> int:
    """Remove tasks in terminal state that are older than max_age_seconds.

    Removes tasks from the registry that have completed (in any terminal state:
    COMPLETED, FAILED, CANCELLED, TIMEOUT) and whose completed_at timestamp
    is older than max_age_seconds. This function is safe to call periodically
    to prevent memory leaks from accumulating completed tasks.

    Args:
        max_age_seconds: Maximum age in seconds for terminal tasks (default 5 minutes).
                        Only tasks older than this value will be removed.

    Returns:
        Number of tasks removed from the registry.
    """
    import time

    global _registry
    now = time.time()

    with _registry_lock:
        stale_ids = [
            task_id
            for task_id, task in _registry.items()
            if task.is_done
            and task.completed_at
            and (now - task.completed_at) > max_age_seconds
        ]
        for task_id in stale_ids:
            del _registry[task_id]
        return len(stale_ids)


async def cleanup_stale_tasks_async(max_age_seconds: float = 300) -> int:
    """Remove tasks in terminal state that are older than max_age_seconds (async version).

    Async-safe version of cleanup_stale_tasks() for use in async contexts.
    Removes tasks from the registry that have completed (in any terminal state:
    COMPLETED, FAILED, CANCELLED, TIMEOUT) and whose completed_at timestamp
    is older than max_age_seconds. This function is safe to call periodically
    to prevent memory leaks from accumulating completed tasks.

    Args:
        max_age_seconds: Maximum age in seconds for terminal tasks (default 5 minutes).
                        Only tasks older than this value will be removed.

    Returns:
        Number of tasks removed from the registry.
    """
    import asyncio

    return await asyncio.to_thread(cleanup_stale_tasks, max_age_seconds)
