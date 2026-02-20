"""
JSON spec file operations for SDD workflows.

I/O functions (find, load, save, backup, list, diff, rollback) live in ``io.py``.
Constants live in ``_constants.py``.
Hierarchy operations (get/update node, phase add/remove/move, recalculate hours)
live in ``hierarchy.py``.
Analysis functions (completeness, duplicate detection) live in ``analysis.py``.
This module contains find-replace operations and will be further split in later phases.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Analysis functions — extracted to analysis.py, re-imported so that
# intra-monolith callers continue to resolve without changes.
from foundry_mcp.core.spec.analysis import (  # noqa: F401 — re-exports
    check_spec_completeness,
    detect_duplicate_tasks,
)

# Hierarchy functions — extracted to hierarchy.py, re-imported so that
# intra-monolith callers continue to resolve without changes.
from foundry_mcp.core.spec.hierarchy import (  # noqa: F401 — re-exports
    _add_phase_verification,
    _collect_descendants,
    _count_tasks_in_subtree,
    _generate_phase_id,
    _normalize_acceptance_criteria,
    _remove_dependency_references,
    add_phase,
    add_phase_bulk,
    get_node,
    move_phase,
    recalculate_actual_hours,
    recalculate_estimated_hours,
    remove_phase,
    update_node,
    update_phase_metadata,
)

# I/O functions — extracted to io.py, re-imported so that intra-monolith
# callers continue to resolve without changes.
from foundry_mcp.core.spec.io import (  # noqa: F401 — re-exports
    _apply_backup_retention,
    _diff_node,
    _load_spec_source,
    _migrate_spec_fields,
    _validate_spec_structure,
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

# Template/creation functions — extracted to templates.py, re-imported so that
# intra-monolith callers continue to resolve without changes.
from foundry_mcp.core.spec.templates import (  # noqa: F401 — re-exports
    add_assumption,
    add_revision,
    apply_phase_template,
    create_spec,
    generate_spec_data,
    get_phase_template_structure,
    get_template_structure,
    list_assumptions,
    update_frontmatter,
)

# Safety constraints for find/replace operations
_FR_MAX_PATTERN_LENGTH = 256
_FR_DEFAULT_MAX_REPLACEMENTS = 1000
_FR_VALID_SCOPES = {"all", "titles", "descriptions"}
_FR_MAX_SAMPLE_DIFFS = 10


def find_replace_in_spec(
    spec_id: str,
    find: str,
    replace: str,
    *,
    scope: str = "all",
    use_regex: bool = False,
    case_sensitive: bool = True,
    dry_run: bool = False,
    max_replacements: int = _FR_DEFAULT_MAX_REPLACEMENTS,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find and replace text across spec hierarchy nodes.

    Performs literal or regex find/replace across titles and/or descriptions
    in a specification's hierarchy nodes.

    Args:
        spec_id: Specification ID to modify.
        find: Text or regex pattern to find.
        replace: Replacement text (supports backreferences if use_regex=True).
        scope: Where to search - "all", "titles", or "descriptions".
        use_regex: If True, treat `find` as a regex pattern.
        case_sensitive: If False, perform case-insensitive matching.
        dry_run: If True, preview changes without modifying the spec.
        max_replacements: Maximum number of replacements (safety limit).
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "total_replacements": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate find pattern
    if not find or not isinstance(find, str):
        return None, "find must be a non-empty string"
    # Don't strip the pattern - use exactly what user provides (whitespace may be intentional)
    if not find.strip():
        return None, "find must be a non-empty string"
    if len(find) > _FR_MAX_PATTERN_LENGTH:
        return None, f"find pattern exceeds maximum length of {_FR_MAX_PATTERN_LENGTH} characters"

    # Validate replace
    if replace is None:
        return None, "replace must be provided (use empty string to delete matches)"
    if not isinstance(replace, str):
        return None, "replace must be a string"

    # Validate scope
    if scope not in _FR_VALID_SCOPES:
        return None, f"scope must be one of: {sorted(_FR_VALID_SCOPES)}"

    # Validate max_replacements
    if not isinstance(max_replacements, int) or max_replacements <= 0:
        return None, "max_replacements must be a positive integer"

    # Compile regex if needed
    compiled_pattern = None
    if use_regex:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled_pattern = re.compile(find, flags)
        except re.error as e:
            return None, f"Invalid regex pattern: {e}"
    else:
        # For literal search, prepare flags
        if not case_sensitive:
            # Create case-insensitive literal pattern
            compiled_pattern = re.compile(re.escape(find), re.IGNORECASE)

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found"

    # Load spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if not spec_path:
        return None, f"Specification '{spec_id}' not found"
    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return {
            "spec_id": spec_id,
            "total_replacements": 0,
            "nodes_affected": 0,
            "changes": [],
            "dry_run": dry_run,
            "message": "No hierarchy nodes to process",
        }, None

    # Track changes
    changes: List[Dict[str, Any]] = []
    total_replacements = 0
    nodes_affected = set()
    warnings: List[str] = []
    limit_reached = False

    # Helper to perform replacement
    def do_replace(text: str) -> Tuple[str, int]:
        if compiled_pattern:
            new_text, count = compiled_pattern.subn(replace, text)
            return new_text, count
        else:
            # Case-sensitive literal replace
            count = text.count(find)
            new_text = text.replace(find, replace)
            return new_text, count

    # Process hierarchy nodes
    for node_id, node_data in hierarchy.items():
        if node_id == "spec-root":
            continue
        if limit_reached:
            break

        # Process title if in scope
        if scope in ("all", "titles"):
            title = node_data.get("title", "")
            if title and isinstance(title, str):
                new_title, count = do_replace(title)
                if count > 0:
                    if total_replacements + count > max_replacements:
                        count = max_replacements - total_replacements
                        # Partial replacement not supported, skip this field
                        warnings.append(f"max_replacements limit ({max_replacements}) reached")
                        limit_reached = True
                    else:
                        total_replacements += count
                        nodes_affected.add(node_id)
                        changes.append(
                            {
                                "node_id": node_id,
                                "field": "title",
                                "old": title,
                                "new": new_title,
                                "replacement_count": count,
                            }
                        )
                        if not dry_run:
                            node_data["title"] = new_title

        # Process description if in scope
        if scope in ("all", "descriptions") and not limit_reached:
            metadata = node_data.get("metadata", {})
            if isinstance(metadata, dict):
                description = metadata.get("description", "")
                if description and isinstance(description, str):
                    new_description, count = do_replace(description)
                    if count > 0:
                        if total_replacements + count > max_replacements:
                            warnings.append(f"max_replacements limit ({max_replacements}) reached")
                            limit_reached = True
                        else:
                            total_replacements += count
                            nodes_affected.add(node_id)
                            changes.append(
                                {
                                    "node_id": node_id,
                                    "field": "description",
                                    "old": description,
                                    "new": new_description,
                                    "replacement_count": count,
                                }
                            )
                            if not dry_run:
                                metadata["description"] = new_description

    # Save if not dry_run and there were changes
    if not dry_run and total_replacements > 0:
        if not save_spec(spec_id, spec_data, specs_dir):
            return None, "Failed to save specification after replacements"

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "total_replacements": total_replacements,
        "nodes_affected": len(nodes_affected),
        "dry_run": dry_run,
        "scope": scope,
        "find": find,
        "replace": replace,
        "use_regex": use_regex,
        "case_sensitive": case_sensitive,
    }

    # Include sample diffs (limited)
    if changes:
        result["changes"] = changes[:_FR_MAX_SAMPLE_DIFFS]
        if len(changes) > _FR_MAX_SAMPLE_DIFFS:
            result["changes_truncated"] = True
            result["total_changes"] = len(changes)

    if warnings:
        result["warnings"] = warnings

    if total_replacements == 0:
        result["message"] = "No matches found"

    return result, None
