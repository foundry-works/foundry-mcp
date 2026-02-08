"""Spec operations package.

Re-exports the full public API so that all existing imports continue to work::

    from foundry_mcp.core.spec import load_spec, save_spec, ...

Sub-modules:
- ``_constants``: Shared constants
- ``io``: I/O functions (find, load, save, backup, list, diff, rollback)
- ``hierarchy``: Hierarchy operations (get/update node, phase CRUD, recalculate hours)
- ``templates``: Spec creation, phase templates, assumptions, revisions, frontmatter
- ``analysis``: Read-only analysis (completeness checks, duplicate detection)
- ``_monolith``: Remaining spec operations (find-replace)
"""

from foundry_mcp.core.spec._constants import (
    CATEGORIES,
    DEFAULT_BACKUP_PAGE_SIZE,
    DEFAULT_DIFF_MAX_RESULTS,
    DEFAULT_MAX_BACKUPS,
    FRONTMATTER_KEYS,
    MAX_BACKUP_PAGE_SIZE,
    PHASE_TEMPLATES,
    TEMPLATES,
    TEMPLATE_DESCRIPTIONS,
    VERIFICATION_TYPES,
)
from foundry_mcp.core.spec.io import (
    _apply_backup_retention,
    backup_spec,
    diff_specs,
    find_git_root,
    find_spec_file,
    find_specs_directory,
    generate_spec_id,
    list_spec_backups,
    list_specs,
    load_spec,
    resolve_spec_file,
    rollback_spec,
    save_spec,
)
from foundry_mcp.core.spec.hierarchy import (
    # Hierarchy/node functions
    get_node,
    update_node,
    # Phase operations
    add_phase,
    add_phase_bulk,
    move_phase,
    recalculate_actual_hours,
    recalculate_estimated_hours,
    remove_phase,
    update_phase_metadata,
)
from foundry_mcp.core.spec.templates import (
    # Spec creation
    create_spec,
    generate_spec_data,
    get_template_structure,
    # Phase template operations
    apply_phase_template,
    get_phase_template_structure,
    # Frontmatter and metadata
    update_frontmatter,
    # Assumptions and revisions
    add_assumption,
    add_revision,
    list_assumptions,
)
from foundry_mcp.core.spec.analysis import (
    # Analysis and validation
    check_spec_completeness,
    detect_duplicate_tasks,
)
from foundry_mcp.core.spec._monolith import (
    # Find and replace
    find_replace_in_spec,
)

__all__ = [
    # Constants
    "CATEGORIES",
    "DEFAULT_BACKUP_PAGE_SIZE",
    "DEFAULT_DIFF_MAX_RESULTS",
    "DEFAULT_MAX_BACKUPS",
    "FRONTMATTER_KEYS",
    "MAX_BACKUP_PAGE_SIZE",
    "PHASE_TEMPLATES",
    "TEMPLATES",
    "TEMPLATE_DESCRIPTIONS",
    "VERIFICATION_TYPES",
    # I/O functions (from io.py)
    "backup_spec",
    "diff_specs",
    "find_git_root",
    "find_spec_file",
    "find_specs_directory",
    "generate_spec_id",
    "list_spec_backups",
    "list_specs",
    "load_spec",
    "resolve_spec_file",
    "rollback_spec",
    "save_spec",
    # Hierarchy/node functions
    "get_node",
    "update_node",
    # Spec creation
    "create_spec",
    "generate_spec_data",
    "get_template_structure",
    # Phase operations
    "add_phase",
    "add_phase_bulk",
    "apply_phase_template",
    "get_phase_template_structure",
    "move_phase",
    "remove_phase",
    "update_phase_metadata",
    # Frontmatter and metadata
    "update_frontmatter",
    # Assumptions and revisions
    "add_assumption",
    "add_revision",
    "list_assumptions",
    # Analysis and validation
    "check_spec_completeness",
    "detect_duplicate_tasks",
    "recalculate_actual_hours",
    "recalculate_estimated_hours",
    # Find and replace
    "find_replace_in_spec",
]
