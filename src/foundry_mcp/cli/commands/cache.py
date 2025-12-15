"""Cache management commands for SDD CLI.

Provides commands for inspecting and managing the AI consultation cache.
"""

from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_success
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.core.cache import CacheManager, is_cache_enabled

logger = get_cli_logger()


@click.group("cache")
def cache() -> None:
    """AI consultation cache management."""
    pass


@cache.command("info")
@click.pass_context
@cli_command("info")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Cache info lookup timed out")
def cache_info_cmd(ctx: click.Context) -> None:
    """Show cache information and statistics.

    Displays cache location, size, and entry counts.
    """
    if not is_cache_enabled():
        emit_success(
            {
                "enabled": False,
                "message": "Cache is disabled",
                "hint": "Unset FOUNDRY_MCP_CACHE_DISABLED to enable caching",
            }
        )
        return

    manager = CacheManager()
    stats = manager.get_stats()

    emit_success(
        {
            "enabled": True,
            **stats,
        }
    )


@cache.command("clear")
@click.option("--spec-id", help="Only clear entries for this spec ID.")
@click.option(
    "--review-type",
    type=click.Choice(["fidelity", "plan"]),
    help="Only clear entries of this review type.",
)
@click.pass_context
@cli_command("clear")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Cache clear timed out")
def cache_clear_cmd(
    ctx: click.Context,
    spec_id: Optional[str],
    review_type: Optional[str],
) -> None:
    """Clear cache entries with optional filters.

    Without filters, clears all cache entries.
    Use --spec-id and/or --review-type to filter.
    """
    if not is_cache_enabled():
        emit_success(
            {
                "enabled": False,
                "entries_deleted": 0,
                "message": "Cache is disabled",
            }
        )
        return

    manager = CacheManager()
    deleted = manager.clear(spec_id=spec_id, review_type=review_type)

    filters = {}
    if spec_id:
        filters["spec_id"] = spec_id
    if review_type:
        filters["review_type"] = review_type

    emit_success(
        {
            "entries_deleted": deleted,
            "filters": filters if filters else None,
        }
    )


@cache.command("cleanup")
@click.pass_context
@cli_command("cleanup")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Cache cleanup timed out")
def cache_cleanup_cmd(ctx: click.Context) -> None:
    """Remove expired cache entries.

    Cleans up entries that have exceeded their TTL.
    """
    if not is_cache_enabled():
        emit_success(
            {
                "enabled": False,
                "entries_removed": 0,
                "message": "Cache is disabled",
            }
        )
        return

    manager = CacheManager()
    removed = manager.cleanup_expired()

    emit_success(
        {
            "entries_removed": removed,
            "message": f"Removed {removed} expired entries"
            if removed
            else "No expired entries found",
        }
    )
