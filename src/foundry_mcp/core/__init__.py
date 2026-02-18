"""Core spec and task operations for foundry-mcp."""

from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    resolve_spec_file,
    load_spec,
    save_spec,
    backup_spec,
    list_specs,
    get_node,
    update_node,
)

from foundry_mcp.core.task import (
    is_unblocked,
    is_in_current_phase,
    get_next_task,
    check_dependencies,
    get_previous_sibling,
    get_parent_context,
    get_phase_context,
    get_task_journal_summary,
    prepare_task,
)

from foundry_mcp.core.progress import (
    recalculate_progress,
    update_node_status,
    update_parent_status,
    get_progress_summary,
    list_phases,
    get_task_counts_by_status,
)

from foundry_mcp.core.validation.models import (
    Diagnostic,
    ValidationResult,
    FixAction,
    FixReport,
    SpecStats,
)
from foundry_mcp.core.validation.rules import validate_spec
from foundry_mcp.core.validation.fixes import get_fix_actions
from foundry_mcp.core.validation.application import apply_fixes
from foundry_mcp.core.validation.stats import calculate_stats

from foundry_mcp.core.journal import (
    JournalEntry,
    BlockerInfo,
    ResolvedBlocker,
    add_journal_entry,
    get_journal_entries,
    get_latest_journal_entry,
    mark_blocked,
    unblock,
    get_blocker_info,
    get_resolved_blockers,
    list_blocked_tasks,
    update_task_status,
    mark_task_journaled,
    find_unjournaled_tasks,
    save_journal,
)

from foundry_mcp.core.responses.types import ToolResponse
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.naming import canonical_tool

from foundry_mcp.core.review import (
    ReviewFinding,
    QuickReviewResult,
    ReviewContext,
    get_llm_status,
    prepare_review_context,
    quick_review,
    review_type_requires_llm,
)

__all__ = [
    "find_specs_directory",
    "find_spec_file",
    "resolve_spec_file",
    "load_spec",
    "save_spec",
    "backup_spec",
    "list_specs",
    "get_node",
    "update_node",
    "is_unblocked",
    "is_in_current_phase",
    "get_next_task",
    "check_dependencies",
    "get_previous_sibling",
    "get_parent_context",
    "get_phase_context",
    "get_task_journal_summary",
    "prepare_task",
    "recalculate_progress",
    "update_node_status",
    "update_parent_status",
    "get_progress_summary",
    "list_phases",
    "get_task_counts_by_status",
    "Diagnostic",
    "ValidationResult",
    "FixAction",
    "FixReport",
    "SpecStats",
    "validate_spec",
    "get_fix_actions",
    "apply_fixes",
    "calculate_stats",
    "JournalEntry",
    "BlockerInfo",
    "ResolvedBlocker",
    "add_journal_entry",
    "get_journal_entries",
    "get_latest_journal_entry",
    "mark_blocked",
    "unblock",
    "get_blocker_info",
    "get_resolved_blockers",
    "list_blocked_tasks",
    "update_task_status",
    "mark_task_journaled",
    "find_unjournaled_tasks",
    "save_journal",
    "ToolResponse",
    "success_response",
    "error_response",
    "canonical_tool",
    # Review
    "ReviewFinding",
    "QuickReviewResult",
    "ReviewContext",
    "get_llm_status",
    "prepare_review_context",
    "quick_review",
    "review_type_requires_llm",
]
