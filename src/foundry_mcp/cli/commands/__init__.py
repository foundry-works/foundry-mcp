"""CLI command groups.

The CLI is organized into domain groups (e.g. `specs`, `tasks`, `test`).
Legacy top-level aliases are intentionally not exported.
"""

from foundry_mcp.cli.commands.cache import cache
from foundry_mcp.cli.commands.dev import dev_group
from foundry_mcp.cli.commands.journal import journal
from foundry_mcp.cli.commands.lifecycle import lifecycle
from foundry_mcp.cli.commands.modify import modify_group
from foundry_mcp.cli.commands.plan import plan_group
from foundry_mcp.cli.commands.review import review_group
from foundry_mcp.cli.commands.session import session
from foundry_mcp.cli.commands.specs import specs
from foundry_mcp.cli.commands.tasks import tasks
from foundry_mcp.cli.commands.validate import validate_group

__all__ = [
    "cache",
    "dev_group",
    "journal",
    "lifecycle",
    "modify_group",
    "plan_group",
    "review_group",
    "session",
    "specs",
    "tasks",
    "validate_group",
]
