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
        cache,
        dashboard_group,
        dev_group,
        journal,
        lifecycle,
        modify_group,
        plan_group,
        pr_group,
        review_group,
        session,
        specs,
        tasks,
        test_group,
        validate_group,
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

    # Placeholder: version command for testing the scaffold
    @cli.command("version")
    @click.pass_context
    def version(ctx: click.Context) -> None:
        """Show CLI version information."""
        from foundry_mcp.cli.output import emit

        cli_ctx = get_context(ctx)
        specs_dir = cli_ctx.specs_dir

        emit(
            {
                "version": "0.1.0",
                "name": "foundry-cli",
                "json_only": True,
                "specs_dir": str(specs_dir) if specs_dir else None,
            }
        )
