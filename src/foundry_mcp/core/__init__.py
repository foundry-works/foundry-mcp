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
]
