"""Command registry for SDD CLI.

Centralized registration of all command groups.
Commands are organized by domain (specs, tasks, journal, etc.).
"""

from typing import Optional

import click

from foundry_mcp.cli.config import CLIContext

# Module-level storage for CLI context (for testing)
_cli_context: Optional[CLIContext] = None


def set_context(ctx: CLIContext) -> None:
    """Set the CLI context at module level.

    Primarily used for testing when not using Click's context.

    Args:
        ctx: The CLIContext to store.
    """
    global _cli_context
    _cli_context = ctx


def get_context(ctx: Optional[click.Context] = None) -> CLIContext:
    """Get CLI context from Click context or module-level storage.

    Args:
        ctx: Optional Click context with cli_context stored in obj.
             If None, returns module-level context.

    Returns:
        The CLIContext instance.

    Raises:
        RuntimeError: If no context is available.
    """
    if ctx is not None:
        return ctx.obj["cli_context"]

    if _cli_context is not None:
        return _cli_context

    raise RuntimeError("No CLI context available. Call set_context() first.")


def register_all_commands(cli: click.Group) -> None:
    """Register all command groups with the CLI.

    Command groups are lazily imported to avoid circular dependencies
    and improve startup time.

    Args:
        cli: The main Click group to register commands with.
    """
    # Import and register command groups
    from foundry_mcp.cli.commands import (
        activate_spec_cmd,
        analyze,
        archive_spec_cmd,
        block_task_cmd,
        cache,
        complete_spec_cmd,
        complete_task_cmd,
        create,
        create_pr_alias_cmd,
        dashboard_group,
        dashboard_start_alias_cmd,
        dev_group,
        dev_install_alias_cmd,
        find_specs_alias_cmd,
        generate_marker_cmd,
        journal,
        journal_add_alias_cmd,
        journal_get_alias_cmd,
        journal_list_alias_cmd,
        journal_unjournaled_alias_cmd,
        lifecycle,
        list_blockers_alias_cmd,
        list_phases_alias_cmd,
        lifecycle_state_cmd,
        modify_apply_cmd,
        modify_group,
        move_spec_cmd,
        next_task,
        plan_group,
        plan_review_alias_cmd,
        pr_group,
        prepare_task_cmd,
        query_tasks_alias_cmd,
        review_group,
        review_spec_alias_cmd,
        run_tests_alias_cmd,
        schema_cmd,
        session,
        session_capabilities_cmd,
        session_status_cmd,
        show_limits_cmd,
        specs,
        task_info_cmd,
        tasks,
        template,
        test_group,
        token_usage_cmd,
        unblock_task_cmd,
        update_status_cmd,
        fix_cmd,
        validate_cmd,
        validate_group,
        work_mode_cmd,
    )

    cli.add_command(specs)
    cli.add_command(tasks)
    cli.add_command(lifecycle)
    cli.add_command(session)
    cli.add_command(cache)
    cli.add_command(journal)
    cli.add_command(validate_group)
    cli.add_command(review_group)
    cli.add_command(pr_group)
    cli.add_command(modify_group)
    cli.add_command(test_group)
    cli.add_command(dev_group)
    cli.add_command(dashboard_group)
    cli.add_command(plan_group)

    # Add top-level aliases for common commands
    # This allows both `sdd specs create` and `sdd create`
    cli.add_command(create, name="create")
    cli.add_command(analyze, name="analyze")
    cli.add_command(template, name="template")
    cli.add_command(next_task, name="next-task")
    cli.add_command(prepare_task_cmd, name="prepare-task")
    cli.add_command(task_info_cmd, name="task-info")
    cli.add_command(update_status_cmd, name="update-status")
    cli.add_command(block_task_cmd, name="block-task")
    cli.add_command(unblock_task_cmd, name="unblock-task")
    cli.add_command(complete_task_cmd, name="complete-task")
    cli.add_command(activate_spec_cmd, name="activate-spec")
    cli.add_command(complete_spec_cmd, name="complete-spec")
    cli.add_command(archive_spec_cmd, name="archive-spec")
    cli.add_command(move_spec_cmd, name="move-spec")
    cli.add_command(lifecycle_state_cmd, name="lifecycle-state")
    cli.add_command(session_status_cmd, name="session-status")
    cli.add_command(show_limits_cmd, name="session-limits")
    cli.add_command(session_capabilities_cmd, name="capabilities")
    cli.add_command(review_spec_alias_cmd, name="review-spec")
    cli.add_command(create_pr_alias_cmd, name="create-pr")
    cli.add_command(modify_apply_cmd, name="apply-modifications")
    cli.add_command(run_tests_alias_cmd, name="run-tests")
    cli.add_command(dev_install_alias_cmd, name="dev-install")
    cli.add_command(journal_add_alias_cmd, name="journal-add")
    cli.add_command(journal_add_alias_cmd, name="add-journal")
    cli.add_command(journal_get_alias_cmd, name="get-journal")
    cli.add_command(journal_list_alias_cmd, name="journal-list")
    cli.add_command(journal_unjournaled_alias_cmd, name="journal-unjournaled")
    cli.add_command(find_specs_alias_cmd, name="find-specs")
    cli.add_command(list_phases_alias_cmd, name="list-phases")
    cli.add_command(query_tasks_alias_cmd, name="query-tasks")
    cli.add_command(list_blockers_alias_cmd, name="list-blockers")
    cli.add_command(fix_cmd, name="fix")
    cli.add_command(schema_cmd, name="schema")
    cli.add_command(plan_review_alias_cmd, name="plan-review")

    # Placeholder: version command for testing the scaffold
    @cli.command("version")
    @click.pass_context
    def version(ctx: click.Context) -> None:
        """Show CLI version information."""
        from foundry_mcp.cli.output import emit

        cli_ctx = get_context(ctx)
        specs_dir = cli_ctx.specs_dir

        emit({
            "version": "0.1.0",
            "name": "foundry-cli",
            "json_only": True,
            "specs_dir": str(specs_dir) if specs_dir else None,
        })
