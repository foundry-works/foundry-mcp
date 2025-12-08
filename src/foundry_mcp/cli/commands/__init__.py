"""CLI command groups for SDD CLI.

This package contains all command group implementations organized by domain.
"""

from foundry_mcp.cli.commands.lifecycle import (
    activate_spec_cmd,
    archive_spec_cmd,
    complete_spec_cmd,
    lifecycle,
    lifecycle_state_cmd,
    move_spec_cmd,
)
from foundry_mcp.cli.commands.session import (
    generate_marker_cmd,
    record_consultation_cmd,
    reset_session_cmd,
    session,
    session_capabilities_cmd,
    session_status_cmd,
    show_limits_cmd,
    start_session_cmd,
    token_usage_cmd,
    work_mode_cmd,
)
from foundry_mcp.cli.commands.specs import (
    analyze,
    create,
    find_specs_cmd,
    find_specs_alias_cmd,
    list_blockers_cmd,
    list_blockers_alias_cmd,
    list_phases_cmd,
    list_phases_alias_cmd,
    query_tasks_cmd,
    query_tasks_alias_cmd,
    schema_cmd,
    specs,
    template,
)
from foundry_mcp.cli.commands.cache import (
    cache,
    cache_clear_cmd,
    cache_cleanup_cmd,
    cache_info_cmd,
)
from foundry_mcp.cli.commands.pr import (
    create_pr_alias_cmd,
    pr_context_cmd,
    pr_create_cmd,
    pr_group,
    pr_status_cmd,
)
from foundry_mcp.cli.commands.render import (
    render_cmd,
    render_group,
)
from foundry_mcp.cli.commands.review import (
    review_fidelity_cmd,
    review_group,
    review_plan_tools_cmd,
    review_spec_alias_cmd,
    review_spec_cmd,
    review_tools_cmd,
)
from foundry_mcp.cli.commands.validate import (
    fix_cmd,
    validate_analyze_deps_cmd,
    validate_cmd,
    validate_group,
    validate_report_cmd,
)
from foundry_mcp.cli.commands.modify import (
    modify_apply_cmd,
    modify_assumption_cmd,
    modify_frontmatter_cmd,
    modify_group,
    modify_revision_cmd,
    modify_task_add_cmd,
    modify_task_group,
    modify_task_remove_cmd,
)
from foundry_mcp.cli.commands.docquery import (
    doc_find_class_cmd,
    doc_find_function_cmd,
    doc_group,
    doc_impact_cmd,
    doc_stats_cmd,
    doc_trace_calls_cmd,
    find_class_alias_cmd,
    find_function_alias_cmd,
)
from foundry_mcp.cli.commands.testing import (
    run_tests_alias_cmd,
    test_check_tools_cmd,
    test_discover_cmd,
    test_group,
    test_presets_cmd,
    test_quick_cmd,
    test_run_cmd,
    test_unit_cmd,
)
from foundry_mcp.cli.commands.docgen import (
    generate_docs_alias_cmd,
    llm_doc_cache_cmd,
    llm_doc_generate_cmd,
    llm_doc_group,
    llm_doc_status_cmd,
)
from foundry_mcp.cli.commands.dev import (
    dev_check_cmd,
    dev_gendocs_cmd,
    dev_group,
    dev_install_alias_cmd,
    dev_install_cmd,
    dev_start_cmd,
)
from foundry_mcp.cli.commands.journal import (
    journal,
    journal_add_cmd,
    journal_add_alias_cmd,
    journal_get_cmd,
    journal_get_alias_cmd,
    journal_list_cmd,
    journal_list_alias_cmd,
    journal_unjournaled_cmd,
    journal_unjournaled_alias_cmd,
)
from foundry_mcp.cli.commands.tasks import (
    block_task_cmd,
    complete_task_cmd,
    next_task,
    prepare_task_cmd,
    task_info_cmd,
    tasks,
    unblock_task_cmd,
    update_status_cmd,
)
from foundry_mcp.cli.commands.dashboard import (
    dashboard_group,
    dashboard_start_alias_cmd,
    dashboard_start_cmd,
    dashboard_status_cmd,
    dashboard_stop_cmd,
)

__all__ = [
    # Spec commands
    "specs",
    "create",
    "analyze",
    "template",
    "schema_cmd",
    "find_specs_cmd",
    "find_specs_alias_cmd",
    "list_phases_cmd",
    "list_phases_alias_cmd",
    "query_tasks_cmd",
    "query_tasks_alias_cmd",
    "list_blockers_cmd",
    "list_blockers_alias_cmd",
    # Task commands
    "tasks",
    "next_task",
    "prepare_task_cmd",
    "task_info_cmd",
    "update_status_cmd",
    "block_task_cmd",
    "unblock_task_cmd",
    "complete_task_cmd",
    # Lifecycle commands
    "lifecycle",
    "activate_spec_cmd",
    "complete_spec_cmd",
    "archive_spec_cmd",
    "move_spec_cmd",
    "lifecycle_state_cmd",
    # Cache commands
    "cache",
    "cache_info_cmd",
    "cache_clear_cmd",
    "cache_cleanup_cmd",
    # Session commands
    "session",
    "start_session_cmd",
    "session_status_cmd",
    "session_capabilities_cmd",
    "record_consultation_cmd",
    "reset_session_cmd",
    "show_limits_cmd",
    "work_mode_cmd",
    "token_usage_cmd",
    "generate_marker_cmd",
    # PR commands
    "create_pr_alias_cmd",
    "pr_context_cmd",
    "pr_create_cmd",
    "pr_group",
    "pr_status_cmd",
    # Render commands
    "render_cmd",
    "render_group",
    # Review commands
    "review_fidelity_cmd",
    "review_group",
    "review_plan_tools_cmd",
    "review_spec_alias_cmd",
    "review_spec_cmd",
    "review_tools_cmd",
    # Validate commands
    "fix_cmd",
    "validate_analyze_deps_cmd",
    "validate_cmd",
    "validate_group",
    "validate_report_cmd",
    # Modify commands
    "modify_apply_cmd",
    "modify_assumption_cmd",
    "modify_frontmatter_cmd",
    "modify_group",
    "modify_revision_cmd",
    "modify_task_add_cmd",
    "modify_task_group",
    "modify_task_remove_cmd",
    # Doc-query commands
    "doc_find_class_cmd",
    "doc_find_function_cmd",
    "doc_group",
    "doc_impact_cmd",
    "doc_stats_cmd",
    "doc_trace_calls_cmd",
    "find_class_alias_cmd",
    "find_function_alias_cmd",
    # Testing commands
    "run_tests_alias_cmd",
    "test_check_tools_cmd",
    "test_discover_cmd",
    "test_group",
    "test_presets_cmd",
    "test_quick_cmd",
    "test_run_cmd",
    "test_unit_cmd",
    # LLM doc generation commands
    "generate_docs_alias_cmd",
    "llm_doc_cache_cmd",
    "llm_doc_generate_cmd",
    "llm_doc_group",
    "llm_doc_status_cmd",
    # Dev utility commands
    "dev_check_cmd",
    "dev_gendocs_cmd",
    "dev_group",
    "dev_install_alias_cmd",
    "dev_install_cmd",
    "dev_start_cmd",
    # Journal commands
    "journal",
    "journal_add_cmd",
    "journal_add_alias_cmd",
    "journal_get_cmd",
    "journal_get_alias_cmd",
    "journal_list_cmd",
    "journal_list_alias_cmd",
    "journal_unjournaled_cmd",
    "journal_unjournaled_alias_cmd",
    # Dashboard commands
    "dashboard_group",
    "dashboard_start_alias_cmd",
    "dashboard_start_cmd",
    "dashboard_status_cmd",
    "dashboard_stop_cmd",
]
