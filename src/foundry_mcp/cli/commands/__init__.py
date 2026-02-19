"""CLI command groups.

The CLI is organized into domain groups (e.g. `specs`, `tasks`, `test`).
Legacy top-level aliases are intentionally not exported.
"""

from foundry_mcp.cli.commands.audit import audit
from foundry_mcp.cli.commands.cache import cache
from foundry_mcp.cli.commands.dev import dev_group
from foundry_mcp.cli.commands.journal import journal
from foundry_mcp.cli.commands.lifecycle import lifecycle
from foundry_mcp.cli.commands.modify import modify_group
from foundry_mcp.cli.commands.plan import plan_group
from foundry_mcp.cli.commands.review import review_group
from foundry_mcp.cli.commands.run import run_cmd
from foundry_mcp.cli.commands.session import session
from foundry_mcp.cli.commands.specs import specs
from foundry_mcp.cli.commands.stop import stop_cmd
from foundry_mcp.cli.commands.tasks import tasks
from foundry_mcp.cli.commands.validate import validate_group
from foundry_mcp.cli.commands.watch import watch_cmd

__all__ = [
    "audit",
    "cache",
    "dev_group",
    "journal",
    "lifecycle",
    "modify_group",
    "plan_group",
    "review_group",
    "run_cmd",
    "session",
    "specs",
    "stop_cmd",
    "tasks",
    "validate_group",
    "watch_cmd",
]
