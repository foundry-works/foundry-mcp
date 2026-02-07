"""Task operations package.

Re-exports the full public API so that all existing imports continue to work::

    from foundry_mcp.core.task import add_task, prepare_task, ...

Sub-modules:
- ``_helpers``: Shared constants and utilities
- ``queries``: Read-only query and context functions
- ``mutations``: Task mutation operations (add, remove, update, move)
- ``batch``: Batch update operations
"""

from foundry_mcp.core.task._helpers import (
    TASK_TYPES,
    REQUIREMENT_TYPES,
)
from foundry_mcp.core.task.queries import (
    check_dependencies,
    get_next_task,
    get_parent_context,
    get_phase_context,
    get_previous_sibling,
    get_task_journal_summary,
    is_in_current_phase,
    is_unblocked,
    prepare_task,
)
from foundry_mcp.core.task.mutations import (
    # Constants
    COMPLEXITY_LEVELS,
    DEPENDENCY_TYPES,
    MAX_REQUIREMENTS_PER_TASK,
    TASK_CATEGORIES,
    VERIFICATION_TYPES,
    # Also re-export CATEGORIES via mutations
    CATEGORIES,
    # Mutation functions
    add_task,
    manage_task_dependency,
    move_task,
    remove_task,
    update_estimate,
    update_task_metadata,
    update_task_requirements,
)
from foundry_mcp.core.task.batch import (
    # Batch constants
    BATCH_ALLOWED_STATUSES,
    DEFAULT_MAX_MATCHES,
    MAX_PATTERN_LENGTH,
    # Batch functions
    batch_update_tasks,
)

__all__ = [
    # Constants
    "BATCH_ALLOWED_STATUSES",
    "CATEGORIES",
    "COMPLEXITY_LEVELS",
    "DEFAULT_MAX_MATCHES",
    "DEPENDENCY_TYPES",
    "MAX_PATTERN_LENGTH",
    "MAX_REQUIREMENTS_PER_TASK",
    "REQUIREMENT_TYPES",
    "TASK_CATEGORIES",
    "TASK_TYPES",
    "VERIFICATION_TYPES",
    # Query functions
    "check_dependencies",
    "get_next_task",
    "get_parent_context",
    "get_phase_context",
    "get_previous_sibling",
    "get_task_journal_summary",
    "is_in_current_phase",
    "is_unblocked",
    "prepare_task",
    # Mutation functions
    "add_task",
    "manage_task_dependency",
    "move_task",
    "remove_task",
    "update_estimate",
    "update_task_metadata",
    "update_task_requirements",
    # Batch functions
    "batch_update_tasks",
]
