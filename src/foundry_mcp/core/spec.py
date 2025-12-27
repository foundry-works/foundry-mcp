"""
JSON spec file operations for SDD workflows.
Provides loading, saving, finding, and listing specs with atomic writes and backups.
"""

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

# Valid templates and categories for spec creation
TEMPLATES = ("empty", "simple", "medium", "complex", "security")
TEMPLATE_DESCRIPTIONS = {
    "empty": "Blank spec with no phases - use for quick tasks or custom structure",
    "simple": "1 phase with basic tasks - use for small features or bug fixes",
    "medium": "3 phases (setup, implementation, verification) - use for standard features",
    "complex": "5+ phases with subtasks and dependencies - use for large features or refactors",
    "security": "Security-focused template with audit phases - use for auth, crypto, or sensitive data",
}
CATEGORIES = ("investigation", "implementation", "refactoring", "decision", "research")

# Valid verification types for verify nodes
# - run-tests: Automated tests via mcp__foundry-mcp__test-run
# - fidelity: Implementation-vs-spec comparison via mcp__foundry-mcp__spec-review-fidelity
# - manual: Manual verification steps
VERIFICATION_TYPES = ("run-tests", "fidelity", "manual")

# Valid phase templates for reusable phase structures
PHASE_TEMPLATES = ("planning", "implementation", "testing", "security", "documentation")


def _requires_rich_task_fields(spec_data: Dict[str, Any]) -> bool:
    metadata = spec_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    template = metadata.get("template")
    if isinstance(template, str) and template.strip().lower() in {"medium", "complex"}:
        return True

    complexity = metadata.get("complexity")
    if isinstance(complexity, str) and complexity.strip().lower() in {
        "medium",
        "complex",
        "high",
    }:
        return True

    return False


def _normalize_acceptance_criteria(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    cleaned_items.append(cleaned)
        return cleaned_items
    return []


def find_git_root() -> Optional[Path]:
    """Find the root of the git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def find_specs_directory(provided_path: Optional[str] = None) -> Optional[Path]:
    """
    Discover the specs directory.

    Args:
        provided_path: Optional explicit path to specs directory or file

    Returns:
        Absolute Path to specs directory (containing pending/active/completed/archived),
        or None if not found
    """

    def is_valid_specs_dir(p: Path) -> bool:
        """Check if a directory is a valid specs directory."""
        return (
            (p / "pending").is_dir()
            or (p / "active").is_dir()
            or (p / "completed").is_dir()
            or (p / "archived").is_dir()
        )

    if provided_path:
        path = Path(provided_path).resolve()

        if path.is_file():
            path = path.parent

        if not path.is_dir():
            return None

        if is_valid_specs_dir(path):
            return path

        specs_subdir = path / "specs"
        if specs_subdir.is_dir() and is_valid_specs_dir(specs_subdir):
            return specs_subdir

        for parent in list(path.parents)[:5]:
            if is_valid_specs_dir(parent):
                return parent
            parent_specs = parent / "specs"
            if parent_specs.is_dir() and is_valid_specs_dir(parent_specs):
                return parent_specs

        return None

    git_root = find_git_root()

    if git_root:
        search_paths = [
            Path.cwd() / "specs",
            git_root / "specs",
        ]
    else:
        search_paths = [
            Path.cwd() / "specs",
            Path.cwd().parent / "specs",
        ]

    for p in search_paths:
        if p.exists() and is_valid_specs_dir(p):
            return p.resolve()

    return None


def find_spec_file(spec_id: str, specs_dir: Path) -> Optional[Path]:
    """
    Find the spec file for a given spec ID.

    Searches in pending/, active/, completed/, and archived/ subdirectories.

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory

    Returns:
        Absolute path to the spec file, or None if not found
    """
    search_dirs = ["pending", "active", "completed", "archived"]

    for subdir in search_dirs:
        spec_file = specs_dir / subdir / f"{spec_id}.json"
        if spec_file.exists():
            return spec_file

    return None


def resolve_spec_file(
    spec_name_or_path: str, specs_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Resolve spec file from either a spec name or full path.

    Args:
        spec_name_or_path: Either a spec name or full path
        specs_dir: Optional specs directory for name-based lookups

    Returns:
        Resolved Path object if found, None otherwise
    """
    path = Path(spec_name_or_path)

    if path.is_absolute():
        spec_file = path.resolve()
        if spec_file.exists() and spec_file.suffix == ".json":
            return spec_file
        return None

    search_name = spec_name_or_path
    if spec_name_or_path.endswith(".json"):
        spec_file = path.resolve()
        if spec_file.exists() and spec_file.suffix == ".json":
            return spec_file
        search_name = path.stem

    if specs_dir is None:
        specs_dir = find_specs_directory()

    if not specs_dir:
        return None

    return find_spec_file(search_name, specs_dir)


def _migrate_spec_fields(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate spec from dual-field format to canonical format.

    Moves status, progress_percentage, and current_phase from metadata
    to top-level (their canonical location). This handles specs created
    before the field deduplication.

    Args:
        spec_data: Spec data dictionary (modified in place)

    Returns:
        The modified spec_data
    """
    if not spec_data:
        return spec_data

    metadata = spec_data.get("metadata", {})
    computed_fields = ("status", "progress_percentage", "current_phase")

    for field in computed_fields:
        # If field exists in metadata but not at top-level, migrate it
        if field in metadata and field not in spec_data:
            spec_data[field] = metadata[field]
        # Remove from metadata (canonical location is top-level)
        metadata.pop(field, None)

    return spec_data


def load_spec(
    spec_id: str, specs_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Load the JSON spec file for a given spec ID or path.

    Args:
        spec_id: Specification ID or path to spec file
        specs_dir: Path to specs directory (optional, auto-detected if not provided)

    Returns:
        Spec data dictionary, or None if not found
    """
    spec_file = resolve_spec_file(spec_id, specs_dir)

    if not spec_file:
        return None

    try:
        with open(spec_file, "r") as f:
            spec_data = json.load(f)
            # Migrate old specs to canonical field locations
            return _migrate_spec_fields(spec_data)
    except (json.JSONDecodeError, IOError):
        return None


def save_spec(
    spec_id: str,
    spec_data: Dict[str, Any],
    specs_dir: Optional[Path] = None,
    backup: bool = True,
    validate: bool = True,
) -> bool:
    """
    Save JSON spec file with atomic write and optional backup.

    Args:
        spec_id: Specification ID or path to spec file
        spec_data: Spec data to write
        specs_dir: Path to specs directory (optional, auto-detected if not provided)
        backup: Create backup before writing (default: True)
        validate: Validate JSON before writing (default: True)

    Returns:
        True if successful, False otherwise
    """
    spec_file = resolve_spec_file(spec_id, specs_dir)

    if not spec_file:
        return False

    if validate:
        if not _validate_spec_structure(spec_data):
            return False

    spec_data["last_updated"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    if backup:
        backup_spec(spec_id, specs_dir)

    temp_file = spec_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w") as f:
            json.dump(spec_data, f, indent=2)
        temp_file.replace(spec_file)
        return True
    except (IOError, OSError):
        if temp_file.exists():
            temp_file.unlink()
        return False


# Default retention policy for versioned backups
DEFAULT_MAX_BACKUPS = 10


def backup_spec(
    spec_id: str,
    specs_dir: Optional[Path] = None,
    max_backups: int = DEFAULT_MAX_BACKUPS,
) -> Optional[Path]:
    """
    Create a versioned backup of the JSON spec file.

    Creates timestamped backups in .backups/{spec_id}/ directory with a
    configurable retention policy. Also maintains a latest.json copy for
    quick access to the most recent backup.

    Directory structure:
        .backups/
          └── {spec_id}/
              ├── 2025-12-26T18-20-13.456789.json   # Timestamped backups (μs precision)
              ├── 2025-12-26T18-30-45.123456.json
              └── latest.json                       # Copy of most recent

    Args:
        spec_id: Specification ID or path to spec file
        specs_dir: Path to specs directory (optional, auto-detected if not provided)
        max_backups: Maximum number of versioned backups to retain (default: 10).
                     Set to 0 for unlimited backups.

    Returns:
        Path to backup file if created, None otherwise
    """
    spec_file = resolve_spec_file(spec_id, specs_dir)

    if not spec_file:
        return None

    if specs_dir is None:
        specs_dir = find_specs_directory()

    if not specs_dir:
        return None

    # Create versioned backup directory: .backups/{spec_id}/
    spec_backups_dir = specs_dir / ".backups" / spec_id
    spec_backups_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp filename (ISO format with safe characters)
    # Include full microseconds to handle rapid successive saves
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    micros = now.strftime("%f")  # Full 6-digit microseconds
    backup_file = spec_backups_dir / f"{timestamp}.{micros}.json"

    try:
        # Create the timestamped backup
        shutil.copy2(spec_file, backup_file)

        # Update latest.json to point to the newest backup
        latest_file = spec_backups_dir / "latest.json"
        shutil.copy2(backup_file, latest_file)

        # Apply retention policy
        if max_backups > 0:
            _apply_backup_retention(spec_backups_dir, max_backups)

        return backup_file
    except (IOError, OSError):
        return None


def _apply_backup_retention(backups_dir: Path, max_backups: int) -> int:
    """
    Apply retention policy by removing oldest backups exceeding the limit.

    Args:
        backups_dir: Path to the spec's backup directory
        max_backups: Maximum number of backups to retain

    Returns:
        Number of backups deleted
    """
    # List all timestamped backup files (exclude latest.json)
    backup_files = sorted(
        [
            f for f in backups_dir.glob("*.json")
            if f.name != "latest.json" and f.is_file()
        ],
        key=lambda p: p.name,  # Sort by filename (timestamp order)
    )

    deleted_count = 0
    while len(backup_files) > max_backups:
        oldest = backup_files.pop(0)
        try:
            oldest.unlink()
            deleted_count += 1
        except (IOError, OSError):
            pass  # Best effort deletion

    return deleted_count


# Default pagination settings for backup listing
DEFAULT_BACKUP_PAGE_SIZE = 50
MAX_BACKUP_PAGE_SIZE = 100


def list_spec_backups(
    spec_id: str,
    specs_dir: Optional[Path] = None,
    cursor: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    List backups for a spec with cursor-based pagination.

    Lists timestamped backup files chronologically (newest first) from the
    .backups/{spec_id}/ directory. Returns file metadata including timestamp,
    path, and size. Designed for use with spec.history action.

    Args:
        spec_id: Specification ID to list backups for
        specs_dir: Base specs directory (uses find_specs_directory if None)
        cursor: Pagination cursor from previous call (base64-encoded JSON)
        limit: Maximum backups per page (default: 50, max: 100)

    Returns:
        Dict with structure:
            {
                "spec_id": str,
                "backups": [
                    {
                        "timestamp": str,      # ISO-ish format from filename
                        "file_path": str,      # Absolute path to backup file
                        "file_size_bytes": int # File size
                    },
                    ...
                ],
                "count": int,
                "pagination": {
                    "cursor": Optional[str],
                    "has_more": bool,
                    "page_size": int
                }
            }

        Returns empty backups list if spec or backup directory doesn't exist.
    """
    # Import pagination helpers
    from foundry_mcp.core.pagination import (
        CursorError,
        decode_cursor,
        encode_cursor,
        normalize_page_size,
    )

    # Resolve specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    # Normalize page size
    page_size = normalize_page_size(
        limit, default=DEFAULT_BACKUP_PAGE_SIZE, maximum=MAX_BACKUP_PAGE_SIZE
    )

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "backups": [],
        "count": 0,
        "pagination": {
            "cursor": None,
            "has_more": False,
            "page_size": page_size,
        },
    }

    if not specs_dir:
        return result

    # Locate backup directory: .backups/{spec_id}/
    backups_dir = specs_dir / ".backups" / spec_id
    if not backups_dir.is_dir():
        return result

    # List all timestamped backup files (exclude latest.json)
    backup_files = sorted(
        [
            f
            for f in backups_dir.glob("*.json")
            if f.name != "latest.json" and f.is_file()
        ],
        key=lambda p: p.name,
        reverse=True,  # Newest first
    )

    if not backup_files:
        return result

    # Handle cursor-based pagination
    start_after_timestamp: Optional[str] = None
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            start_after_timestamp = cursor_data.get("last_id")
        except CursorError:
            # Invalid cursor - return from beginning
            pass

    # Find start position based on cursor
    if start_after_timestamp:
        start_index = 0
        for idx, backup_file in enumerate(backup_files):
            # Filename without extension is the timestamp
            timestamp = backup_file.stem
            if timestamp == start_after_timestamp:
                start_index = idx + 1
                break
        backup_files = backup_files[start_index:]

    # Fetch one extra to check for more pages
    page_files = backup_files[: page_size + 1]
    has_more = len(page_files) > page_size
    if has_more:
        page_files = page_files[:page_size]

    # Build backup entries with metadata
    backups = []
    for backup_file in page_files:
        try:
            file_stat = backup_file.stat()
            backups.append(
                {
                    "timestamp": backup_file.stem,
                    "file_path": str(backup_file.absolute()),
                    "file_size_bytes": file_stat.st_size,
                }
            )
        except OSError:
            # Skip files we can't stat
            continue

    # Generate next cursor if more pages exist
    next_cursor = None
    if has_more and backups:
        next_cursor = encode_cursor({"last_id": backups[-1]["timestamp"]})

    result["backups"] = backups
    result["count"] = len(backups)
    result["pagination"] = {
        "cursor": next_cursor,
        "has_more": has_more,
        "page_size": page_size,
    }

    return result


# Default settings for diff operations
DEFAULT_DIFF_MAX_RESULTS = 100


def _load_spec_source(
    source: Union[str, Path, Dict[str, Any]],
    specs_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load a spec from various source types.

    Args:
        source: Spec ID, file path, or already-loaded dict
        specs_dir: Base specs directory for ID lookups

    Returns:
        Loaded spec dict, or None if not found/invalid
    """
    # Already a dict - return as-is
    if isinstance(source, dict):
        return source

    # Path object or string path
    source_path = Path(source) if isinstance(source, str) else source

    # If it's an existing file path, load directly
    if source_path.is_file():
        try:
            with open(source_path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

    # Otherwise treat as spec_id and use resolve_spec_file
    if isinstance(source, str):
        return load_spec(source, specs_dir)

    return None


def _diff_node(
    old_node: Dict[str, Any],
    new_node: Dict[str, Any],
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Compare two nodes and return field-level changes.

    Args:
        old_node: Original node data
        new_node: Updated node data
        node_id: Node identifier for the result

    Returns:
        Dict with node info and field_changes list, or None if no changes
    """
    # Fields to compare (excluding computed/transient fields)
    compare_fields = ["title", "status", "type", "parent", "children", "metadata", "dependencies"]

    field_changes = []
    for field in compare_fields:
        old_val = old_node.get(field)
        new_val = new_node.get(field)

        if old_val != new_val:
            field_changes.append({
                "field": field,
                "old": old_val,
                "new": new_val,
            })

    if not field_changes:
        return None

    return {
        "node_id": node_id,
        "type": new_node.get("type", old_node.get("type")),
        "title": new_node.get("title", old_node.get("title")),
        "field_changes": field_changes,
    }


def diff_specs(
    source: Union[str, Path, Dict[str, Any]],
    target: Union[str, Path, Dict[str, Any]],
    specs_dir: Optional[Path] = None,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare two specs and categorize changes as added, removed, or modified.

    Compares hierarchy nodes between source (base/older) and target (comparison/newer)
    specs, identifying structural and content changes at the task level.

    Args:
        source: Base spec - spec_id, file path (including backup), or loaded dict
        target: Comparison spec - spec_id, file path, or loaded dict
        specs_dir: Base specs directory (auto-detected if None)
        max_results: Maximum changes to return per category (default: 100)

    Returns:
        Dict with structure:
            {
                "summary": {
                    "added_count": int,
                    "removed_count": int,
                    "modified_count": int,
                    "total_changes": int
                },
                "changes": {
                    "added": [{"node_id": str, "type": str, "title": str}, ...],
                    "removed": [{"node_id": str, "type": str, "title": str}, ...],
                    "modified": [{
                        "node_id": str,
                        "type": str,
                        "title": str,
                        "field_changes": [{"field": str, "old": Any, "new": Any}, ...]
                    }, ...]
                },
                "partial": bool,  # True if results truncated
                "source_spec_id": Optional[str],
                "target_spec_id": Optional[str]
            }

        Returns error structure if specs cannot be loaded:
            {"error": str, "success": False}
    """
    # Resolve specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    # Load source spec
    source_spec = _load_spec_source(source, specs_dir)
    if source_spec is None:
        return {
            "error": f"Could not load source spec: {source}",
            "success": False,
        }

    # Load target spec
    target_spec = _load_spec_source(target, specs_dir)
    if target_spec is None:
        return {
            "error": f"Could not load target spec: {target}",
            "success": False,
        }

    # Get hierarchies
    source_hierarchy = source_spec.get("hierarchy", {})
    target_hierarchy = target_spec.get("hierarchy", {})

    source_ids = set(source_hierarchy.keys())
    target_ids = set(target_hierarchy.keys())

    # Categorize changes
    added_ids = target_ids - source_ids
    removed_ids = source_ids - target_ids
    common_ids = source_ids & target_ids

    # Apply max_results limit
    limit = max_results if max_results is not None else DEFAULT_DIFF_MAX_RESULTS
    partial = False

    # Build added list
    added = []
    for node_id in sorted(added_ids):
        if len(added) >= limit:
            partial = True
            break
        node = target_hierarchy[node_id]
        added.append({
            "node_id": node_id,
            "type": node.get("type"),
            "title": node.get("title"),
        })

    # Build removed list
    removed = []
    for node_id in sorted(removed_ids):
        if len(removed) >= limit:
            partial = True
            break
        node = source_hierarchy[node_id]
        removed.append({
            "node_id": node_id,
            "type": node.get("type"),
            "title": node.get("title"),
        })

    # Build modified list
    modified = []
    for node_id in sorted(common_ids):
        if len(modified) >= limit:
            partial = True
            break
        old_node = source_hierarchy[node_id]
        new_node = target_hierarchy[node_id]
        diff = _diff_node(old_node, new_node, node_id)
        if diff:
            modified.append(diff)

    # Calculate actual counts (may exceed displayed if partial)
    total_added = len(added_ids)
    total_removed = len(removed_ids)
    total_modified = sum(
        1 for nid in common_ids
        if _diff_node(source_hierarchy[nid], target_hierarchy[nid], nid)
    ) if not partial else len(modified)  # Only count all if not already partial

    return {
        "summary": {
            "added_count": total_added,
            "removed_count": total_removed,
            "modified_count": total_modified if not partial else len(modified),
            "total_changes": total_added + total_removed + (total_modified if not partial else len(modified)),
        },
        "changes": {
            "added": added,
            "removed": removed,
            "modified": modified,
        },
        "partial": partial,
        "source_spec_id": source_spec.get("spec_id"),
        "target_spec_id": target_spec.get("spec_id"),
    }


def rollback_spec(
    spec_id: str,
    timestamp: str,
    specs_dir: Optional[Path] = None,
    dry_run: bool = False,
    create_backup: bool = True,
) -> Dict[str, Any]:
    """
    Restore a spec from a specific backup timestamp.

    Creates a safety backup of the current state before rollback (by default),
    then replaces the spec file with the contents from the specified backup.

    Args:
        spec_id: Specification ID to rollback
        timestamp: Backup timestamp to restore (e.g., "2025-12-26T18-20-13.456789")
        specs_dir: Base specs directory (auto-detected if None)
        dry_run: If True, validate and return what would happen without changes
        create_backup: If True (default), create safety backup before rollback

    Returns:
        Dict with structure:
            {
                "success": bool,
                "spec_id": str,
                "timestamp": str,
                "dry_run": bool,
                "backup_created": Optional[str],  # Safety backup path
                "restored_from": str,              # Source backup path
                "error": Optional[str]             # Error if failed
            }
    """
    # Resolve specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    result: Dict[str, Any] = {
        "success": False,
        "spec_id": spec_id,
        "timestamp": timestamp,
        "dry_run": dry_run,
        "backup_created": None,
        "restored_from": None,
        "error": None,
    }

    if not specs_dir:
        result["error"] = "Could not find specs directory"
        return result

    # Find current spec file
    spec_file = find_spec_file(spec_id, specs_dir)
    if not spec_file:
        result["error"] = f"Spec '{spec_id}' not found"
        return result

    # Locate backup directory
    backups_dir = specs_dir / ".backups" / spec_id
    if not backups_dir.is_dir():
        result["error"] = f"No backups directory for spec '{spec_id}'"
        return result

    # Find the backup file matching the timestamp
    backup_file = backups_dir / f"{timestamp}.json"
    if not backup_file.is_file():
        result["error"] = f"Backup not found for timestamp '{timestamp}'"
        return result

    result["restored_from"] = str(backup_file)

    # Validate backup is valid JSON
    try:
        with open(backup_file, "r") as f:
            backup_data = json.load(f)
        if not isinstance(backup_data, dict):
            result["error"] = "Backup file is not a valid spec (not a JSON object)"
            return result
    except json.JSONDecodeError as e:
        result["error"] = f"Backup file is not valid JSON: {e}"
        return result
    except IOError as e:
        result["error"] = f"Could not read backup file: {e}"
        return result

    # dry_run - return success without making changes
    if dry_run:
        result["success"] = True
        if create_backup:
            result["backup_created"] = "(would be created)"
        return result

    # Create safety backup of current state before rollback
    if create_backup:
        safety_backup = backup_spec(spec_id, specs_dir)
        if safety_backup:
            result["backup_created"] = str(safety_backup)

    # Perform rollback - copy backup to spec location
    try:
        shutil.copy2(backup_file, spec_file)
        result["success"] = True
    except (IOError, OSError) as e:
        result["error"] = f"Failed to restore backup: {e}"
        return result

    return result


def _validate_spec_structure(spec_data: Dict[str, Any]) -> bool:
    """
    Validate basic JSON spec file structure.

    Args:
        spec_data: Spec data dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["spec_id", "hierarchy"]
    for field in required_fields:
        if field not in spec_data:
            return False

    hierarchy = spec_data.get("hierarchy", {})
    if not isinstance(hierarchy, dict):
        return False

    for node_id, node_data in hierarchy.items():
        if not isinstance(node_data, dict):
            return False
        if "type" not in node_data or "status" not in node_data:
            return False
        if node_data["status"] not in [
            "pending",
            "in_progress",
            "completed",
            "blocked",
        ]:
            return False

    return True


def list_specs(
    specs_dir: Optional[Path] = None, status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List specification files with optional filtering.

    Args:
        specs_dir: Base specs directory (auto-detected if not provided)
        status: Filter by status folder (active, completed, archived, pending, or None for all)

    Returns:
        List of spec info dictionaries
    """
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if not specs_dir:
        return []

    if status and status != "all":
        status_dirs = [specs_dir / status]
    else:
        status_dirs = [
            specs_dir / "active",
            specs_dir / "completed",
            specs_dir / "archived",
            specs_dir / "pending",
        ]

    specs_info = []

    for status_dir in status_dirs:
        if not status_dir.exists():
            continue

        status_name = status_dir.name

        json_files = sorted(status_dir.glob("*.json"))

        for json_file in json_files:
            spec_data = load_spec(json_file.stem, specs_dir)
            if not spec_data:
                continue

            metadata = spec_data.get("metadata", {})
            hierarchy = spec_data.get("hierarchy", {})

            total_tasks = len(hierarchy)
            completed_tasks = sum(
                1 for task in hierarchy.values() if task.get("status") == "completed"
            )

            progress_pct = 0
            if total_tasks > 0:
                progress_pct = int((completed_tasks / total_tasks) * 100)

            info = {
                "spec_id": json_file.stem,
                "status": status_name,
                "title": metadata.get("title", spec_data.get("title", "Untitled")),
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "progress_percentage": progress_pct,
                "current_phase": metadata.get("current_phase"),
            }

            specs_info.append(info)

    # Sort: active first, then by completion % (highest first)
    specs_info.sort(
        key=lambda s: (
            0 if s.get("status") == "active" else 1,
            -s.get("progress_percentage", 0),
        )
    )

    return specs_info


def get_node(spec_data: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific node from the hierarchy.

    Args:
        spec_data: JSON spec file data
        node_id: Node identifier

    Returns:
        Node data dictionary or None if not found
    """
    hierarchy = spec_data.get("hierarchy", {})
    return hierarchy.get(node_id)


def update_node(
    spec_data: Dict[str, Any], node_id: str, updates: Dict[str, Any]
) -> bool:
    """
    Update a node in the hierarchy.

    Special handling for metadata: existing metadata fields are preserved
    and merged with new metadata fields.

    Args:
        spec_data: JSON spec file data
        node_id: Node identifier
        updates: Dictionary of fields to update

    Returns:
        True if node exists and was updated, False otherwise
    """
    hierarchy = spec_data.get("hierarchy", {})

    if node_id not in hierarchy:
        return False

    node = hierarchy[node_id]

    if "metadata" in updates:
        existing_metadata = node.get("metadata", {})
        new_metadata = updates["metadata"]
        updates = updates.copy()
        updates["metadata"] = {**existing_metadata, **new_metadata}

    node.update(updates)
    return True


# =============================================================================
# Spec Creation Functions
# =============================================================================


def generate_spec_id(name: str) -> str:
    """
    Generate a spec ID from a human-readable name.

    Args:
        name: Human-readable spec name.

    Returns:
        URL-safe spec ID with date suffix (e.g., "my-feature-2025-01-15-001").
    """
    # Normalize: lowercase, replace spaces/special chars with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    # Add date suffix
    date_suffix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Add sequence number (001 for new specs)
    return f"{slug}-{date_suffix}-001"


def _add_phase_verification(
    hierarchy: Dict[str, Any], phase_num: int, phase_id: str
) -> None:
    """
    Add verify nodes (auto + fidelity) to a phase.

    Args:
        hierarchy: The hierarchy dict to modify.
        phase_num: Phase number (1, 2, 3, etc.).
        phase_id: Phase node ID (e.g., "phase-1").
    """
    verify_auto_id = f"verify-{phase_num}-1"
    verify_fidelity_id = f"verify-{phase_num}-2"

    # Run tests verification
    hierarchy[verify_auto_id] = {
        "type": "verify",
        "title": "Run tests",
        "status": "pending",
        "parent": phase_id,
        "children": [],
        "total_tasks": 1,
        "completed_tasks": 0,
        "metadata": {
            "verification_type": "run-tests",
            "mcp_tool": "mcp__foundry-mcp__test-run",
            "expected": "All tests pass",
        },
        "dependencies": {
            "blocks": [verify_fidelity_id],
            "blocked_by": [],
            "depends": [],
        },
    }

    # Fidelity verification (spec review)
    hierarchy[verify_fidelity_id] = {
        "type": "verify",
        "title": "Fidelity review",
        "status": "pending",
        "parent": phase_id,
        "children": [],
        "total_tasks": 1,
        "completed_tasks": 0,
        "metadata": {
            "verification_type": "fidelity",
            "mcp_tool": "mcp__foundry-mcp__spec-review-fidelity",
            "scope": "phase",
            "target": phase_id,
            "expected": "Implementation matches specification",
        },
        "dependencies": {
            "blocks": [],
            "blocked_by": [verify_auto_id],
            "depends": [],
        },
    }

    # Update phase children and task count
    hierarchy[phase_id]["children"].extend([verify_auto_id, verify_fidelity_id])
    hierarchy[phase_id]["total_tasks"] += 2


def _generate_phase_id(hierarchy: Dict[str, Any]) -> Tuple[str, int]:
    """Generate the next phase ID and numeric suffix."""
    pattern = re.compile(r"^phase-(\d+)$")
    max_id = 0
    for node_id in hierarchy.keys():
        match = pattern.match(node_id)
        if match:
            max_id = max(max_id, int(match.group(1)))
    next_id = max_id + 1
    return f"phase-{next_id}", next_id


def add_phase(
    spec_id: str,
    title: str,
    description: Optional[str] = None,
    purpose: Optional[str] = None,
    estimated_hours: Optional[float] = None,
    position: Optional[int] = None,
    link_previous: bool = True,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a new phase under spec-root and scaffold verification tasks.

    Args:
        spec_id: Specification ID to mutate.
        title: Phase title.
        description: Optional phase description.
        purpose: Optional purpose/goal metadata string.
        estimated_hours: Optional estimated hours for the phase.
        position: Optional zero-based insertion index in spec-root children.
        link_previous: Whether to automatically block on the previous phase when appending.
        specs_dir: Specs directory override.

    Returns:
        Tuple of (result_dict, error_message).
    """
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not title or not title.strip():
        return None, "Phase title is required"

    if estimated_hours is not None and estimated_hours < 0:
        return None, "estimated_hours must be non-negative"

    title = title.strip()

    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root")

    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    if spec_root.get("type") not in {"spec", "root"}:
        return None, "Specification root node has invalid type"

    children = spec_root.get("children", []) or []
    if not isinstance(children, list):
        children = []

    insert_index = len(children)
    if position is not None and position >= 0:
        insert_index = min(position, len(children))

    phase_id, phase_num = _generate_phase_id(hierarchy)

    metadata: Dict[str, Any] = {
        "purpose": (purpose.strip() if purpose else ""),
    }
    if description:
        metadata["description"] = description.strip()
    if estimated_hours is not None:
        metadata["estimated_hours"] = estimated_hours

    phase_node = {
        "type": "phase",
        "title": title,
        "status": "pending",
        "parent": "spec-root",
        "children": [],
        "total_tasks": 0,
        "completed_tasks": 0,
        "metadata": metadata,
        "dependencies": {
            "blocks": [],
            "blocked_by": [],
            "depends": [],
        },
    }

    hierarchy[phase_id] = phase_node

    if insert_index == len(children):
        children.append(phase_id)
    else:
        children.insert(insert_index, phase_id)
    spec_root["children"] = children

    linked_phase_id: Optional[str] = None
    if link_previous and insert_index > 0 and insert_index == len(children) - 1:
        candidate = children[insert_index - 1]
        previous = hierarchy.get(candidate)
        if previous and previous.get("type") == "phase":
            linked_phase_id = candidate
            prev_deps = previous.setdefault(
                "dependencies",
                {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            )
            blocks = prev_deps.setdefault("blocks", [])
            if phase_id not in blocks:
                blocks.append(phase_id)
            phase_node["dependencies"]["blocked_by"].append(candidate)

    _add_phase_verification(hierarchy, phase_num, phase_id)

    phase_task_total = phase_node.get("total_tasks", 0)
    total_tasks = spec_root.get("total_tasks", 0)
    spec_root["total_tasks"] = total_tasks + phase_task_total

    # Update spec-level estimated hours if provided
    if estimated_hours is not None:
        spec_metadata = spec_data.setdefault("metadata", {})
        current_hours = spec_metadata.get("estimated_hours")
        if isinstance(current_hours, (int, float)):
            spec_metadata["estimated_hours"] = current_hours + estimated_hours
        else:
            spec_metadata["estimated_hours"] = estimated_hours

    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    verify_ids = [f"verify-{phase_num}-1", f"verify-{phase_num}-2"]

    return {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "title": title,
        "position": insert_index,
        "linked_previous": linked_phase_id,
        "verify_tasks": verify_ids,
    }, None


def add_phase_bulk(
    spec_id: str,
    phase_title: str,
    tasks: List[Dict[str, Any]],
    phase_description: Optional[str] = None,
    phase_purpose: Optional[str] = None,
    phase_estimated_hours: Optional[float] = None,
    metadata_defaults: Optional[Dict[str, Any]] = None,
    position: Optional[int] = None,
    link_previous: bool = True,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a new phase with pre-defined tasks in a single atomic operation.

    Creates a phase and all specified tasks/verify nodes without auto-generating
    verification scaffolding. This enables creating complete phase structures
    in one operation.

    Args:
        spec_id: Specification ID to mutate.
        phase_title: Phase title.
        tasks: List of task definitions, each containing:
            - type: "task" or "verify" (required)
            - title: Task title (required)
            - description: Optional description
            - acceptance_criteria: Optional list of acceptance criteria
            - task_category: Optional task category
            - file_path: Optional associated file path
            - estimated_hours: Optional time estimate
            - verification_type: Optional verification type for verify tasks
        phase_description: Optional phase description.
        phase_purpose: Optional purpose/goal metadata string.
        phase_estimated_hours: Optional estimated hours for the phase.
        metadata_defaults: Optional defaults applied to tasks missing explicit values.
            Supported keys: task_category, category, acceptance_criteria, estimated_hours
        position: Optional zero-based insertion index in spec-root children.
        link_previous: Whether to automatically block on the previous phase.
        specs_dir: Specs directory override.

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"phase_id": ..., "tasks_created": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate required parameters
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not phase_title or not phase_title.strip():
        return None, "Phase title is required"

    if not tasks or not isinstance(tasks, list) or len(tasks) == 0:
        return None, "At least one task definition is required"

    if phase_estimated_hours is not None and phase_estimated_hours < 0:
        return None, "phase_estimated_hours must be non-negative"

    phase_title = phase_title.strip()
    defaults = metadata_defaults or {}

    # Validate metadata_defaults values
    if defaults:
        default_est_hours = defaults.get("estimated_hours")
        if default_est_hours is not None:
            if not isinstance(default_est_hours, (int, float)) or default_est_hours < 0:
                return None, "metadata_defaults.estimated_hours must be a non-negative number"
        default_category = defaults.get("task_category")
        if default_category is None:
            default_category = defaults.get("category")
        if default_category is not None and not isinstance(default_category, str):
            return None, "metadata_defaults.task_category must be a string"
        default_acceptance = defaults.get("acceptance_criteria")
        if default_acceptance is not None and not isinstance(
            default_acceptance, (list, str)
        ):
            return None, "metadata_defaults.acceptance_criteria must be a list of strings"
        if isinstance(default_acceptance, list) and any(
            not isinstance(item, str) for item in default_acceptance
        ):
            return None, "metadata_defaults.acceptance_criteria must be a list of strings"

    # Validate each task definition
    valid_task_types = {"task", "verify"}
    for idx, task_def in enumerate(tasks):
        if not isinstance(task_def, dict):
            return None, f"Task at index {idx} must be a dictionary"

        task_type = task_def.get("type")
        if not task_type or task_type not in valid_task_types:
            return None, f"Task at index {idx} must have type 'task' or 'verify'"

        task_title = task_def.get("title")
        if not task_title or not isinstance(task_title, str) or not task_title.strip():
            return None, f"Task at index {idx} must have a non-empty title"

        est_hours = task_def.get("estimated_hours")
        if est_hours is not None:
            if not isinstance(est_hours, (int, float)) or est_hours < 0:
                return None, f"Task at index {idx} has invalid estimated_hours"

        task_category = task_def.get("task_category")
        if task_category is not None and not isinstance(task_category, str):
            return None, f"Task at index {idx} has invalid task_category"

        legacy_category = task_def.get("category")
        if legacy_category is not None and not isinstance(legacy_category, str):
            return None, f"Task at index {idx} has invalid category"

        acceptance_criteria = task_def.get("acceptance_criteria")
        if acceptance_criteria is not None and not isinstance(
            acceptance_criteria, (list, str)
        ):
            return None, f"Task at index {idx} has invalid acceptance_criteria"
        if isinstance(acceptance_criteria, list) and any(
            not isinstance(item, str) for item in acceptance_criteria
        ):
            return None, f"Task at index {idx} acceptance_criteria must be a list of strings"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    requires_rich_tasks = _requires_rich_task_fields(spec_data)

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root")

    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    if spec_root.get("type") not in {"spec", "root"}:
        return None, "Specification root node has invalid type"

    children = spec_root.get("children", []) or []
    if not isinstance(children, list):
        children = []

    insert_index = len(children)
    if position is not None and position >= 0:
        insert_index = min(position, len(children))

    # Generate phase ID
    phase_id, phase_num = _generate_phase_id(hierarchy)

    # Build phase metadata
    phase_metadata: Dict[str, Any] = {
        "purpose": (phase_purpose.strip() if phase_purpose else ""),
    }
    if phase_description:
        phase_metadata["description"] = phase_description.strip()
    if phase_estimated_hours is not None:
        phase_metadata["estimated_hours"] = phase_estimated_hours

    # Create phase node (without children initially)
    phase_node = {
        "type": "phase",
        "title": phase_title,
        "status": "pending",
        "parent": "spec-root",
        "children": [],
        "total_tasks": 0,
        "completed_tasks": 0,
        "metadata": phase_metadata,
        "dependencies": {
            "blocks": [],
            "blocked_by": [],
            "depends": [],
        },
    }

    hierarchy[phase_id] = phase_node

    # Insert phase into spec-root children
    if insert_index == len(children):
        children.append(phase_id)
    else:
        children.insert(insert_index, phase_id)
    spec_root["children"] = children

    # Link to previous phase if requested
    linked_phase_id: Optional[str] = None
    if link_previous and insert_index > 0 and insert_index == len(children) - 1:
        candidate = children[insert_index - 1]
        previous = hierarchy.get(candidate)
        if previous and previous.get("type") == "phase":
            linked_phase_id = candidate
            prev_deps = previous.setdefault(
                "dependencies",
                {"blocks": [], "blocked_by": [], "depends": []},
            )
            blocks = prev_deps.setdefault("blocks", [])
            if phase_id not in blocks:
                blocks.append(phase_id)
            phase_node["dependencies"]["blocked_by"].append(candidate)

    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _extract_description(task_def: Dict[str, Any]) -> tuple[Optional[str], Any]:
        description = task_def.get("description")
        if _nonempty_string(description) and isinstance(description, str):
            return "description", description.strip()
        details = task_def.get("details")
        if _nonempty_string(details) and isinstance(details, str):
            return "details", details.strip()
        if isinstance(details, list):
            cleaned = [
                item.strip()
                for item in details
                if isinstance(item, str) and item.strip()
            ]
            if cleaned:
                return "details", cleaned
        return None, None

    # Create tasks under the phase
    tasks_created: List[Dict[str, Any]] = []
    task_counter = 0
    verify_counter = 0

    for task_def in tasks:
        task_type = task_def["type"]
        task_title = task_def["title"].strip()

        # Generate task ID based on type
        if task_type == "verify":
            verify_counter += 1
            task_id = f"verify-{phase_num}-{verify_counter}"
        else:
            task_counter += 1
            task_id = f"task-{phase_num}-{task_counter}"

        # Build task metadata with defaults cascade
        task_metadata: Dict[str, Any] = {}

        # Apply description/details
        desc_field, desc_value = _extract_description(task_def)
        if desc_field and desc_value is not None:
            task_metadata[desc_field] = desc_value
        elif requires_rich_tasks and task_type == "task":
            return None, f"Task '{task_title}' missing description"

        # Apply file_path
        file_path = task_def.get("file_path")
        if file_path and isinstance(file_path, str):
            task_metadata["file_path"] = file_path.strip()

        # Apply estimated_hours (task-level overrides defaults)
        est_hours = task_def.get("estimated_hours")
        if est_hours is not None:
            task_metadata["estimated_hours"] = float(est_hours)
        elif defaults.get("estimated_hours") is not None:
            task_metadata["estimated_hours"] = float(defaults["estimated_hours"])

        normalized_category = None
        if task_type == "task":
            # Apply acceptance_criteria
            raw_acceptance = task_def.get("acceptance_criteria")
            if raw_acceptance is None:
                raw_acceptance = defaults.get("acceptance_criteria")
            acceptance_criteria = _normalize_acceptance_criteria(raw_acceptance)
            if acceptance_criteria is not None:
                task_metadata["acceptance_criteria"] = acceptance_criteria
            if requires_rich_tasks:
                if raw_acceptance is None:
                    return None, f"Task '{task_title}' missing acceptance_criteria"
                if not acceptance_criteria:
                    return (
                        None,
                        f"Task '{task_title}' acceptance_criteria must include at least one entry",
                    )

            # Apply task_category from defaults if not specified
            category = task_def.get("task_category") or task_def.get("category")
            if category is None:
                category = defaults.get("task_category") or defaults.get("category")
            if category and isinstance(category, str):
                normalized_category = category.strip().lower()
                if normalized_category not in CATEGORIES:
                    return (
                        None,
                        f"Task '{task_title}' has invalid task_category '{category}'",
                    )
                task_metadata["task_category"] = normalized_category
            if requires_rich_tasks and normalized_category is None:
                return None, f"Task '{task_title}' missing task_category"

            if normalized_category in {"implementation", "refactoring"}:
                if not _nonempty_string(task_metadata.get("file_path")):
                    return (
                        None,
                        f"Task '{task_title}' missing file_path for category '{normalized_category}'",
                    )

        # Apply verification_type for verify tasks
        if task_type == "verify":
            verify_type = task_def.get("verification_type")
            if verify_type and verify_type in VERIFICATION_TYPES:
                task_metadata["verification_type"] = verify_type

        # Create task node
        task_node = {
            "type": task_type,
            "title": task_title,
            "status": "pending",
            "parent": phase_id,
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": task_metadata,
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }

        hierarchy[task_id] = task_node
        phase_node["children"].append(task_id)
        phase_node["total_tasks"] += 1

        tasks_created.append({
            "task_id": task_id,
            "title": task_title,
            "type": task_type,
        })

    # Update spec-root total_tasks
    total_tasks = spec_root.get("total_tasks", 0)
    spec_root["total_tasks"] = total_tasks + phase_node["total_tasks"]

    # Update spec-level estimated hours if provided
    if phase_estimated_hours is not None:
        spec_metadata = spec_data.setdefault("metadata", {})
        current_hours = spec_metadata.get("estimated_hours")
        if isinstance(current_hours, (int, float)):
            spec_metadata["estimated_hours"] = current_hours + phase_estimated_hours
        else:
            spec_metadata["estimated_hours"] = phase_estimated_hours

    # Save spec atomically
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "title": phase_title,
        "position": insert_index,
        "linked_previous": linked_phase_id,
        "tasks_created": tasks_created,
        "total_tasks": len(tasks_created),
    }, None


def _collect_descendants(hierarchy: Dict[str, Any], node_id: str) -> List[str]:
    """
    Recursively collect all descendant node IDs for a given node.

    Args:
        hierarchy: The spec hierarchy dict
        node_id: Starting node ID

    Returns:
        List of all descendant node IDs (not including the starting node)
    """
    descendants: List[str] = []
    node = hierarchy.get(node_id)
    if not node:
        return descendants

    children = node.get("children", [])
    if not isinstance(children, list):
        return descendants

    for child_id in children:
        descendants.append(child_id)
        descendants.extend(_collect_descendants(hierarchy, child_id))

    return descendants


def _count_tasks_in_subtree(
    hierarchy: Dict[str, Any], node_ids: List[str]
) -> Tuple[int, int]:
    """
    Count total and completed tasks in a list of nodes.

    Args:
        hierarchy: The spec hierarchy dict
        node_ids: List of node IDs to count

    Returns:
        Tuple of (total_count, completed_count)
    """
    total = 0
    completed = 0

    for node_id in node_ids:
        node = hierarchy.get(node_id)
        if not node:
            continue
        node_type = node.get("type")
        if node_type in ("task", "subtask", "verify"):
            total += 1
            if node.get("status") == "completed":
                completed += 1

    return total, completed


def _remove_dependency_references(
    hierarchy: Dict[str, Any], removed_ids: List[str]
) -> None:
    """
    Remove references to deleted nodes from all dependency lists.

    Args:
        hierarchy: The spec hierarchy dict
        removed_ids: List of node IDs being removed
    """
    removed_set = set(removed_ids)

    for node_id, node in hierarchy.items():
        deps = node.get("dependencies")
        if not deps or not isinstance(deps, dict):
            continue

        for key in ("blocks", "blocked_by", "depends"):
            dep_list = deps.get(key)
            if isinstance(dep_list, list):
                deps[key] = [d for d in dep_list if d not in removed_set]


def remove_phase(
    spec_id: str,
    phase_id: str,
    force: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Remove a phase and all its children from a specification.

    Handles adjacent phase re-linking: if phase B is removed and A blocks B
    which blocks C, then A will be updated to block C directly.

    Args:
        spec_id: Specification ID containing the phase.
        phase_id: Phase ID to remove (e.g., "phase-1").
        force: If True, remove even if phase contains non-completed tasks.
               If False (default), refuse to remove phases with active work.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phase_id": ..., "children_removed": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate inputs
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not phase_id or not phase_id.strip():
        return None, "Phase ID is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate phase exists
    phase = hierarchy.get(phase_id)
    if phase is None:
        return None, f"Phase '{phase_id}' not found"

    # Validate node type is phase
    node_type = phase.get("type")
    if node_type != "phase":
        return None, f"Node '{phase_id}' is not a phase (type: {node_type})"

    # Collect all descendants
    descendants = _collect_descendants(hierarchy, phase_id)

    # Check for non-completed tasks if force is False
    if not force:
        # Count tasks in phase (excluding verify nodes for the active work check)
        all_nodes = [phase_id] + descendants
        has_active_work = False
        active_task_ids: List[str] = []

        for node_id in all_nodes:
            node = hierarchy.get(node_id)
            if not node:
                continue
            node_status = node.get("status")
            node_node_type = node.get("type")
            # Consider in_progress or pending tasks as active work
            if node_node_type in ("task", "subtask") and node_status in (
                "pending",
                "in_progress",
            ):
                has_active_work = True
                active_task_ids.append(node_id)

        if has_active_work:
            return (
                None,
                f"Phase '{phase_id}' has {len(active_task_ids)} non-completed task(s). "
                f"Use force=True to remove anyway. Active tasks: {', '.join(active_task_ids[:5])}"
                + ("..." if len(active_task_ids) > 5 else ""),
            )

    # Get spec-root and phase position info for re-linking
    spec_root = hierarchy.get("spec-root")
    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    children = spec_root.get("children", [])
    if not isinstance(children, list):
        children = []

    # Find phase position
    try:
        phase_index = children.index(phase_id)
    except ValueError:
        return None, f"Phase '{phase_id}' not found in spec-root children"

    # Identify adjacent phases for re-linking
    prev_phase_id: Optional[str] = None
    next_phase_id: Optional[str] = None

    if phase_index > 0:
        candidate = children[phase_index - 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            prev_phase_id = candidate

    if phase_index < len(children) - 1:
        candidate = children[phase_index + 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            next_phase_id = candidate

    # Re-link adjacent phases: if prev blocks this phase and this phase blocks next,
    # then prev should now block next directly
    relinked_from: Optional[str] = None
    relinked_to: Optional[str] = None

    if prev_phase_id and next_phase_id:
        prev_phase = hierarchy.get(prev_phase_id)
        next_phase = hierarchy.get(next_phase_id)

        if prev_phase and next_phase:
            # Check if prev_phase blocks this phase
            prev_deps = prev_phase.get("dependencies", {})
            prev_blocks = prev_deps.get("blocks", [])

            # Check if this phase blocks next_phase
            phase_deps = phase.get("dependencies", {})
            phase_blocks = phase_deps.get("blocks", [])

            if phase_id in prev_blocks and next_phase_id in phase_blocks:
                # Re-link: prev should now block next
                if next_phase_id not in prev_blocks:
                    prev_blocks.append(next_phase_id)

                # Update next phase's blocked_by
                next_deps = next_phase.setdefault(
                    "dependencies",
                    {
                        "blocks": [],
                        "blocked_by": [],
                        "depends": [],
                    },
                )
                next_blocked_by = next_deps.setdefault("blocked_by", [])
                if prev_phase_id not in next_blocked_by:
                    next_blocked_by.append(prev_phase_id)

                relinked_from = prev_phase_id
                relinked_to = next_phase_id

    # Count tasks being removed
    nodes_to_remove = [phase_id] + descendants
    total_removed, completed_removed = _count_tasks_in_subtree(hierarchy, descendants)

    # Remove all nodes from hierarchy
    for node_id in nodes_to_remove:
        if node_id in hierarchy:
            del hierarchy[node_id]

    # Remove phase from spec-root children
    children.remove(phase_id)
    spec_root["children"] = children

    # Update spec-root task counts
    current_total = spec_root.get("total_tasks", 0)
    current_completed = spec_root.get("completed_tasks", 0)
    spec_root["total_tasks"] = max(0, current_total - total_removed)
    spec_root["completed_tasks"] = max(0, current_completed - completed_removed)

    # Clean up dependency references to removed nodes
    _remove_dependency_references(hierarchy, nodes_to_remove)

    # Save the spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "phase_title": phase.get("title", ""),
        "children_removed": len(descendants),
        "total_tasks_removed": total_removed,
        "completed_tasks_removed": completed_removed,
        "force": force,
    }

    if relinked_from and relinked_to:
        result["relinked"] = {
            "from": relinked_from,
            "to": relinked_to,
        }

    return result, None


def move_phase(
    spec_id: str,
    phase_id: str,
    position: int,
    link_previous: bool = True,
    dry_run: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Move a phase to a new position within spec-root's children.

    Supports reordering phases and optionally re-linking phase dependencies
    according to the link_previous pattern (each phase blocked by its predecessor).

    Args:
        spec_id: Specification ID containing the phase.
        phase_id: Phase ID to move (e.g., "phase-2").
        position: Target position (1-based index) in spec-root children.
        link_previous: If True, update dependencies to maintain the sequential
                       blocking pattern. If False, preserve existing dependencies.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phase_id": ..., "old_position": ..., "new_position": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate inputs
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    if not phase_id or not phase_id.strip():
        return None, "Phase ID is required"

    if not isinstance(position, int) or position < 1:
        return None, "Position must be a positive integer (1-based)"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate phase exists
    phase = hierarchy.get(phase_id)
    if phase is None:
        return None, f"Phase '{phase_id}' not found"

    # Validate node type is phase
    node_type = phase.get("type")
    if node_type != "phase":
        return None, f"Node '{phase_id}' is not a phase (type: {node_type})"

    # Get spec-root
    spec_root = hierarchy.get("spec-root")
    if spec_root is None:
        return None, "Specification root node 'spec-root' not found"

    children = spec_root.get("children", [])
    if not isinstance(children, list):
        children = []

    # Find current position
    try:
        old_index = children.index(phase_id)
    except ValueError:
        return None, f"Phase '{phase_id}' not found in spec-root children"

    # Convert to 0-based index for internal use
    new_index = position - 1

    # Validate position is within bounds
    if new_index < 0 or new_index >= len(children):
        return None, f"Invalid position {position}. Must be 1-{len(children)}"

    # No change needed if same position
    if old_index == new_index:
        return {
            "spec_id": spec_id,
            "phase_id": phase_id,
            "phase_title": phase.get("title", ""),
            "old_position": old_index + 1,
            "new_position": new_index + 1,
            "moved": False,
            "dry_run": dry_run,
            "message": "Phase is already at the specified position",
        }, None

    # Identify old neighbors for dependency cleanup
    old_prev_id: Optional[str] = None
    old_next_id: Optional[str] = None

    if old_index > 0:
        candidate = children[old_index - 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            old_prev_id = candidate

    if old_index < len(children) - 1:
        candidate = children[old_index + 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            old_next_id = candidate

    # Perform the move in children list
    children.remove(phase_id)
    # After removal, adjust target index if moving forward
    insert_index = new_index if new_index <= old_index else new_index
    if insert_index >= len(children):
        children.append(phase_id)
    else:
        children.insert(insert_index, phase_id)

    # Identify new neighbors
    actual_new_index = children.index(phase_id)
    new_prev_id: Optional[str] = None
    new_next_id: Optional[str] = None

    if actual_new_index > 0:
        candidate = children[actual_new_index - 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            new_prev_id = candidate

    if actual_new_index < len(children) - 1:
        candidate = children[actual_new_index + 1]
        if hierarchy.get(candidate, {}).get("type") == "phase":
            new_next_id = candidate

    # Track dependency changes
    dependencies_updated: List[Dict[str, Any]] = []

    if link_previous:
        # Remove old dependency links
        phase_deps = phase.setdefault(
            "dependencies", {"blocks": [], "blocked_by": [], "depends": []}
        )

        # 1. Remove this phase from old_prev's blocks list
        if old_prev_id:
            old_prev = hierarchy.get(old_prev_id)
            if old_prev:
                old_prev_deps = old_prev.get("dependencies", {})
                old_prev_blocks = old_prev_deps.get("blocks", [])
                if phase_id in old_prev_blocks:
                    old_prev_blocks.remove(phase_id)
                    dependencies_updated.append({
                        "action": "removed",
                        "from": old_prev_id,
                        "relationship": "blocks",
                        "target": phase_id,
                    })

        # 2. Remove old_prev from this phase's blocked_by
        phase_blocked_by = phase_deps.setdefault("blocked_by", [])
        if old_prev_id and old_prev_id in phase_blocked_by:
            phase_blocked_by.remove(old_prev_id)
            dependencies_updated.append({
                "action": "removed",
                "from": phase_id,
                "relationship": "blocked_by",
                "target": old_prev_id,
            })

        # 3. Remove this phase from old_next's blocked_by
        if old_next_id:
            old_next = hierarchy.get(old_next_id)
            if old_next:
                old_next_deps = old_next.get("dependencies", {})
                old_next_blocked_by = old_next_deps.get("blocked_by", [])
                if phase_id in old_next_blocked_by:
                    old_next_blocked_by.remove(phase_id)
                    dependencies_updated.append({
                        "action": "removed",
                        "from": old_next_id,
                        "relationship": "blocked_by",
                        "target": phase_id,
                    })

        # 4. Remove old_next from this phase's blocks
        phase_blocks = phase_deps.setdefault("blocks", [])
        if old_next_id and old_next_id in phase_blocks:
            phase_blocks.remove(old_next_id)
            dependencies_updated.append({
                "action": "removed",
                "from": phase_id,
                "relationship": "blocks",
                "target": old_next_id,
            })

        # 5. Link old neighbors to each other (if they were adjacent via this phase)
        if old_prev_id and old_next_id:
            old_prev = hierarchy.get(old_prev_id)
            old_next = hierarchy.get(old_next_id)
            if old_prev and old_next:
                old_prev_deps = old_prev.setdefault(
                    "dependencies", {"blocks": [], "blocked_by": [], "depends": []}
                )
                old_prev_blocks = old_prev_deps.setdefault("blocks", [])
                if old_next_id not in old_prev_blocks:
                    old_prev_blocks.append(old_next_id)
                    dependencies_updated.append({
                        "action": "added",
                        "from": old_prev_id,
                        "relationship": "blocks",
                        "target": old_next_id,
                    })

                old_next_deps = old_next.setdefault(
                    "dependencies", {"blocks": [], "blocked_by": [], "depends": []}
                )
                old_next_blocked_by = old_next_deps.setdefault("blocked_by", [])
                if old_prev_id not in old_next_blocked_by:
                    old_next_blocked_by.append(old_prev_id)
                    dependencies_updated.append({
                        "action": "added",
                        "from": old_next_id,
                        "relationship": "blocked_by",
                        "target": old_prev_id,
                    })

        # Add new dependency links
        # 6. New prev blocks this phase
        if new_prev_id:
            new_prev = hierarchy.get(new_prev_id)
            if new_prev:
                new_prev_deps = new_prev.setdefault(
                    "dependencies", {"blocks": [], "blocked_by": [], "depends": []}
                )
                new_prev_blocks = new_prev_deps.setdefault("blocks", [])
                if phase_id not in new_prev_blocks:
                    new_prev_blocks.append(phase_id)
                    dependencies_updated.append({
                        "action": "added",
                        "from": new_prev_id,
                        "relationship": "blocks",
                        "target": phase_id,
                    })

                # This phase is blocked by new prev
                if new_prev_id not in phase_blocked_by:
                    phase_blocked_by.append(new_prev_id)
                    dependencies_updated.append({
                        "action": "added",
                        "from": phase_id,
                        "relationship": "blocked_by",
                        "target": new_prev_id,
                    })

        # 7. This phase blocks new next
        if new_next_id:
            new_next = hierarchy.get(new_next_id)
            if new_next:
                if new_next_id not in phase_blocks:
                    phase_blocks.append(new_next_id)
                    dependencies_updated.append({
                        "action": "added",
                        "from": phase_id,
                        "relationship": "blocks",
                        "target": new_next_id,
                    })

                new_next_deps = new_next.setdefault(
                    "dependencies", {"blocks": [], "blocked_by": [], "depends": []}
                )
                new_next_blocked_by = new_next_deps.setdefault("blocked_by", [])
                if phase_id not in new_next_blocked_by:
                    new_next_blocked_by.append(phase_id)
                    dependencies_updated.append({
                        "action": "added",
                        "from": new_next_id,
                        "relationship": "blocked_by",
                        "target": phase_id,
                    })

                # Remove old link from new prev to new next (now goes through this phase)
                if new_prev_id:
                    new_prev = hierarchy.get(new_prev_id)
                    if new_prev:
                        new_prev_deps = new_prev.get("dependencies", {})
                        new_prev_blocks = new_prev_deps.get("blocks", [])
                        if new_next_id in new_prev_blocks:
                            new_prev_blocks.remove(new_next_id)
                            dependencies_updated.append({
                                "action": "removed",
                                "from": new_prev_id,
                                "relationship": "blocks",
                                "target": new_next_id,
                            })

                    if new_prev_id in new_next_blocked_by:
                        new_next_blocked_by.remove(new_prev_id)
                        dependencies_updated.append({
                            "action": "removed",
                            "from": new_next_id,
                            "relationship": "blocked_by",
                            "target": new_prev_id,
                        })

    # Update spec-root children
    spec_root["children"] = children

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "phase_title": phase.get("title", ""),
        "old_position": old_index + 1,
        "new_position": actual_new_index + 1,
        "moved": True,
        "link_previous": link_previous,
        "dry_run": dry_run,
    }

    if dependencies_updated:
        result["dependencies_updated"] = dependencies_updated

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        return result, None

    # Save the spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return result, None


def update_phase_metadata(
    spec_id: str,
    phase_id: str,
    *,
    estimated_hours: Optional[float] = None,
    description: Optional[str] = None,
    purpose: Optional[str] = None,
    dry_run: bool = False,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update metadata fields of a phase in a specification.

    Allows updating phase-level metadata such as estimated_hours, description,
    and purpose. Tracks previous values for audit purposes.

    Args:
        spec_id: Specification ID containing the phase.
        phase_id: Phase ID to update (e.g., "phase-1").
        estimated_hours: New estimated hours value (must be >= 0 if provided).
        description: New description text for the phase.
        purpose: New purpose text for the phase.
        dry_run: If True, validate and return preview without saving changes.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "phase_id": ..., "updates": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate spec_id
    if not spec_id or not spec_id.strip():
        return None, "Specification ID is required"

    # Validate phase_id
    if not phase_id or not phase_id.strip():
        return None, "Phase ID is required"

    # Validate estimated_hours if provided
    if estimated_hours is not None:
        if not isinstance(estimated_hours, (int, float)):
            return None, "estimated_hours must be a number"
        if estimated_hours < 0:
            return None, "estimated_hours must be >= 0"

    # Check that at least one field is being updated
    has_update = any(
        v is not None for v in [estimated_hours, description, purpose]
    )
    if not has_update:
        return None, "At least one field (estimated_hours, description, purpose) must be provided"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    hierarchy = spec_data.get("hierarchy", {})

    # Validate phase exists
    phase = hierarchy.get(phase_id)
    if phase is None:
        return None, f"Phase '{phase_id}' not found"

    # Validate node type is phase
    node_type = phase.get("type")
    if node_type != "phase":
        return None, f"Node '{phase_id}' is not a phase (type: {node_type})"

    # Ensure metadata exists on phase
    if "metadata" not in phase:
        phase["metadata"] = {}

    phase_metadata = phase["metadata"]

    # Track updates with previous values
    updates: List[Dict[str, Any]] = []

    if estimated_hours is not None:
        previous = phase_metadata.get("estimated_hours")
        phase_metadata["estimated_hours"] = estimated_hours
        updates.append({
            "field": "estimated_hours",
            "previous_value": previous,
            "new_value": estimated_hours,
        })

    if description is not None:
        description = description.strip() if description else description
        previous = phase_metadata.get("description")
        phase_metadata["description"] = description
        updates.append({
            "field": "description",
            "previous_value": previous,
            "new_value": description,
        })

    if purpose is not None:
        purpose = purpose.strip() if purpose else purpose
        previous = phase_metadata.get("purpose")
        phase_metadata["purpose"] = purpose
        updates.append({
            "field": "purpose",
            "previous_value": previous,
            "new_value": purpose,
        })

    # Build result
    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "phase_title": phase.get("title", ""),
        "updates": updates,
        "dry_run": dry_run,
    }

    if dry_run:
        result["message"] = "Dry run - changes not saved"
        return result, None

    # Save the spec
    saved = save_spec(spec_id, spec_data, specs_dir)
    if not saved:
        return None, "Failed to save specification"

    return result, None


def get_template_structure(template: str, category: str) -> Dict[str, Any]:
    """
    Get the hierarchical structure for a spec template.

    All templates include per-phase verification (auto + fidelity) for each phase.

    Args:
        template: Template type (empty, simple, medium, complex, security).
        category: Default task category.

    Returns:
        Hierarchy dict for the spec.
    """
    # Empty template: just spec-root with no phases (for custom phase-add-bulk workflow)
    if template == "empty":
        return {
            "spec-root": {
                "type": "spec",
                "title": "",  # Filled in later
                "status": "pending",
                "parent": None,
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {
                    "purpose": "",
                    "category": category,
                },
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
        }

    base_hierarchy = {
        "spec-root": {
            "type": "spec",
            "title": "",  # Filled in later
            "status": "pending",
            "parent": None,
            "children": ["phase-1"],
            "total_tasks": 0,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "",
                "category": category,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        },
        "phase-1": {
            "type": "phase",
            "title": "Planning & Discovery",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-1-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Initial planning and requirements gathering",
                "estimated_hours": 2,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        },
        "task-1-1": {
            "type": "task",
            "title": "Define requirements",
            "status": "pending",
            "parent": "phase-1",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "description": "Document the requirements and acceptance criteria",
                "acceptance_criteria": [
                    "Requirements and acceptance criteria are documented",
                ],
                "task_category": "investigation",
                "estimated_hours": 1,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        },
    }

    # Add verification to phase-1 (all templates)
    _add_phase_verification(base_hierarchy, 1, "phase-1")
    base_hierarchy["spec-root"]["total_tasks"] = 3  # task + 2 verify

    if template == "simple":
        return base_hierarchy

    # Medium/complex/security: add implementation phase
    if template in ("medium", "complex", "security"):
        base_hierarchy["spec-root"]["children"].append("phase-2")
        base_hierarchy["phase-1"]["dependencies"]["blocks"].append("phase-2")
        base_hierarchy["phase-2"] = {
            "type": "phase",
            "title": "Implementation",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-2-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Core implementation work",
                "estimated_hours": 8,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": ["phase-1"],
                "depends": [],
            },
        }
        base_hierarchy["task-2-1"] = {
            "type": "task",
            "title": "Implement core functionality",
            "status": "pending",
            "parent": "phase-2",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "description": "Implement the main features",
                "acceptance_criteria": [
                    "Core functionality is implemented and behaves as specified",
                ],
                "task_category": "investigation",
                "estimated_hours": 4,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }
        # Add verification to phase-2
        _add_phase_verification(base_hierarchy, 2, "phase-2")
        base_hierarchy["spec-root"]["total_tasks"] = 6  # 2 tasks + 4 verify

    # Security: add security review phase
    if template == "security":
        base_hierarchy["spec-root"]["children"].append("phase-3")
        base_hierarchy["phase-2"]["dependencies"]["blocks"].append("phase-3")
        base_hierarchy["phase-3"] = {
            "type": "phase",
            "title": "Security Review",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-3-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Security audit and hardening",
                "estimated_hours": 4,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": ["phase-2"],
                "depends": [],
            },
        }
        base_hierarchy["task-3-1"] = {
            "type": "task",
            "title": "Security audit",
            "status": "pending",
            "parent": "phase-3",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "description": "Review for security vulnerabilities",
                "acceptance_criteria": [
                    "Security risks are identified and documented",
                ],
                "task_category": "investigation",
                "estimated_hours": 2,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }
        # Add verification to phase-3
        _add_phase_verification(base_hierarchy, 3, "phase-3")
        base_hierarchy["spec-root"]["total_tasks"] = 9  # 3 tasks + 6 verify

    return base_hierarchy


def get_phase_template_structure(
    template: str, category: str = "implementation"
) -> Dict[str, Any]:
    """
    Get the structure definition for a phase template.

    Phase templates define reusable phase structures with pre-configured tasks.
    Each template includes automatic verification scaffolding (run-tests + fidelity).

    Args:
        template: Phase template type (planning, implementation, testing, security, documentation).
        category: Default task category for tasks in this phase.

    Returns:
        Dict with phase structure including:
        - title: Phase title
        - description: Phase description
        - purpose: Phase purpose for metadata
        - estimated_hours: Total estimated hours
        - tasks: List of task definitions (title, description, category, estimated_hours)
        - includes_verification: Always True (verification auto-added)
    """
    templates: Dict[str, Dict[str, Any]] = {
        "planning": {
            "title": "Planning & Discovery",
            "description": "Requirements gathering, analysis, and initial planning",
            "purpose": "Define scope, requirements, and acceptance criteria",
            "estimated_hours": 4,
            "tasks": [
                {
                    "title": "Define requirements",
                    "description": "Document functional and non-functional requirements",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Requirements are documented and reviewed",
                    ],
                    "estimated_hours": 2,
                },
                {
                    "title": "Design solution approach",
                    "description": "Outline the technical approach and architecture decisions",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Solution approach and key decisions are documented",
                    ],
                    "estimated_hours": 2,
                },
            ],
        },
        "implementation": {
            "title": "Implementation",
            "description": "Core development and feature implementation",
            "purpose": "Build the primary functionality",
            "estimated_hours": 8,
            "tasks": [
                {
                    "title": "Implement core functionality",
                    "description": "Build the main features and business logic",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Core functionality is implemented and verified",
                    ],
                    "estimated_hours": 6,
                },
                {
                    "title": "Add error handling",
                    "description": "Implement error handling and edge cases",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Error handling covers expected edge cases",
                    ],
                    "estimated_hours": 2,
                },
            ],
        },
        "testing": {
            "title": "Testing & Validation",
            "description": "Comprehensive testing and quality assurance",
            "purpose": "Ensure code quality and correctness",
            "estimated_hours": 6,
            "tasks": [
                {
                    "title": "Write unit tests",
                    "description": "Create unit tests for individual components",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Unit tests cover primary logic paths",
                    ],
                    "estimated_hours": 3,
                },
                {
                    "title": "Write integration tests",
                    "description": "Create integration tests for component interactions",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Integration tests cover critical workflows",
                    ],
                    "estimated_hours": 3,
                },
            ],
        },
        "security": {
            "title": "Security Review",
            "description": "Security audit, vulnerability assessment, and hardening",
            "purpose": "Identify and remediate security vulnerabilities",
            "estimated_hours": 6,
            "tasks": [
                {
                    "title": "Security audit",
                    "description": "Review code for security vulnerabilities (OWASP Top 10)",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Security findings are documented with severity",
                    ],
                    "estimated_hours": 3,
                },
                {
                    "title": "Security remediation",
                    "description": "Fix identified vulnerabilities and harden implementation",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Security findings are addressed or tracked",
                    ],
                    "estimated_hours": 3,
                },
            ],
        },
        "documentation": {
            "title": "Documentation",
            "description": "Technical documentation and knowledge capture",
            "purpose": "Document the implementation for maintainability",
            "estimated_hours": 4,
            "tasks": [
                {
                    "title": "Write API documentation",
                    "description": "Document public APIs, parameters, and return values",
                    "task_category": "research",
                    "acceptance_criteria": [
                        "API documentation is updated with current behavior",
                    ],
                    "estimated_hours": 2,
                },
                {
                    "title": "Write user guide",
                    "description": "Create usage examples and integration guide",
                    "task_category": "research",
                    "acceptance_criteria": [
                        "User guide includes usage examples",
                    ],
                    "estimated_hours": 2,
                },
            ],
        },
    }

    if template not in templates:
        raise ValueError(
            f"Invalid phase template '{template}'. Must be one of: {', '.join(PHASE_TEMPLATES)}"
        )

    result = templates[template].copy()
    result["includes_verification"] = True
    result["template_name"] = template
    return result


def apply_phase_template(
    spec_id: str,
    template: str,
    specs_dir: Optional[Path] = None,
    category: str = "implementation",
    position: Optional[int] = None,
    link_previous: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Apply a phase template to an existing spec.

    Creates a new phase with pre-configured tasks based on the template.
    Automatically includes verification scaffolding (run-tests + fidelity).

    Args:
        spec_id: ID of the spec to add the phase to.
        template: Phase template name (planning, implementation, testing, security, documentation).
        specs_dir: Path to specs directory (auto-detected if not provided).
        category: Default task category for tasks (can be overridden by template).
        position: Position to insert phase (None = append at end).
        link_previous: Whether to link this phase to the previous one with dependencies.

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"phase_id": ..., "tasks_created": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate template
    if template not in PHASE_TEMPLATES:
        return (
            None,
            f"Invalid phase template '{template}'. Must be one of: {', '.join(PHASE_TEMPLATES)}",
        )

    # Get template structure
    template_struct = get_phase_template_structure(template, category)

    # Build tasks list for add_phase_bulk
    tasks = []
    for task_def in template_struct["tasks"]:
        tasks.append({
            "type": "task",
            "title": task_def["title"],
            "description": task_def.get("description", ""),
            "task_category": task_def.get("task_category", task_def.get("category", category)),
            "acceptance_criteria": task_def.get("acceptance_criteria"),
            "estimated_hours": task_def.get("estimated_hours", 1),
        })

    # Append verification scaffolding (run-tests + fidelity-review)
    tasks.append({
        "type": "verify",
        "title": "Run tests",
        "verification_type": "run-tests",
    })
    tasks.append({
        "type": "verify",
        "title": "Fidelity review",
        "verification_type": "fidelity",
    })

    # Use add_phase_bulk to create the phase atomically
    result, error = add_phase_bulk(
        spec_id=spec_id,
        phase_title=template_struct["title"],
        tasks=tasks,
        specs_dir=specs_dir,
        phase_description=template_struct.get("description"),
        phase_purpose=template_struct.get("purpose"),
        phase_estimated_hours=template_struct.get("estimated_hours"),
        position=position,
        link_previous=link_previous,
    )

    if error:
        return None, error

    # Enhance result with template info
    if result:
        result["template_applied"] = template
        result["template_title"] = template_struct["title"]

    return result, None


def generate_spec_data(
    name: str,
    template: str = "medium",
    category: str = "implementation",
    mission: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate spec data structure without writing to disk.

    Used for preflight validation (dry_run) and by create_spec.

    Args:
        name: Human-readable name for the specification.
        template: Template type (empty, simple, medium, complex, security).
        category: Default task category.
        mission: Optional mission statement for the spec.

    Returns:
        Tuple of (spec_data, error_message).
        On success: (dict, None)
        On failure: (None, "error message")
    """
    # Validate template
    if template not in TEMPLATES:
        return (
            None,
            f"Invalid template '{template}'. Must be one of: {', '.join(TEMPLATES)}",
        )

    # Validate category
    if category not in CATEGORIES:
        return (
            None,
            f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}",
        )

    if template in ("medium", "complex"):
        if not isinstance(mission, str) or not mission.strip():
            return None, "mission is required for medium/complex specifications"

    # Generate spec ID
    spec_id = generate_spec_id(name)

    # Generate spec structure
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    hierarchy = get_template_structure(template, category)

    # Fill in the title
    hierarchy["spec-root"]["title"] = name

    # Calculate estimated hours from hierarchy
    estimated_hours = sum(
        node.get("metadata", {}).get("estimated_hours", 0)
        for node in hierarchy.values()
        if isinstance(node, dict)
    )

    # Determine current_phase (None for empty template with no phases)
    current_phase = None if template == "empty" else "phase-1"

    spec_data = {
        "spec_id": spec_id,
        "title": name,
        "generated": now,
        "last_updated": now,
        "metadata": {
            "description": "",
            "mission": mission.strip() if isinstance(mission, str) else "",
            "objectives": [],
            "complexity": "medium" if template in ("medium", "complex") else "low",
            "estimated_hours": estimated_hours,
            "assumptions": [],
            "owner": "",
            "category": category,
            "template": template,
        },
        "progress_percentage": 0,
        "status": "pending",
        "current_phase": current_phase,
        "hierarchy": hierarchy,
        "journal": [],
    }

    return spec_data, None


def create_spec(
    name: str,
    template: str = "medium",
    category: str = "implementation",
    mission: Optional[str] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Create a new specification file from a template.

    Args:
        name: Human-readable name for the specification.
        template: Template type (simple, medium, complex, security). Default: medium.
        category: Default task category. Default: implementation.
        mission: Optional mission statement for the spec (required for medium/complex).
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "spec_path": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Generate spec data (handles validation)
    spec_data, error = generate_spec_data(
        name=name,
        template=template,
        category=category,
        mission=mission,
    )
    if error or spec_data is None:
        return None, error or "Failed to generate spec data"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Ensure pending directory exists
    pending_dir = specs_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Check if spec already exists
    spec_id = spec_data["spec_id"]
    spec_path = pending_dir / f"{spec_id}.json"
    if spec_path.exists():
        return None, f"Specification already exists: {spec_id}"

    # Write the spec file
    try:
        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=2)
    except (IOError, OSError) as e:
        return None, f"Failed to write spec file: {e}"

    # Count tasks and phases
    hierarchy = spec_data["hierarchy"]
    task_count = sum(
        1
        for node in hierarchy.values()
        if isinstance(node, dict) and node.get("type") in ("task", "subtask", "verify")
    )
    phase_count = sum(
        1
        for node in hierarchy.values()
        if isinstance(node, dict) and node.get("type") == "phase"
    )

    return {
        "spec_id": spec_id,
        "spec_path": str(spec_path),
        "template": template,
        "category": category,
        "name": name,
        "structure": {
            "phases": phase_count,
            "tasks": task_count,
        },
    }, None


# Valid assumption types
ASSUMPTION_TYPES = ("constraint", "requirement")


def add_assumption(
    spec_id: str,
    text: str,
    assumption_type: str = "constraint",
    author: Optional[str] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add an assumption to a specification's assumptions array.

    The schema expects assumptions to be stored as strings. The assumption_type
    and author are included in the returned result for API compatibility but
    are not stored in the spec (the text itself should be descriptive).

    Args:
        spec_id: Specification ID to add assumption to.
        text: Assumption text/description.
        assumption_type: Type of assumption (constraint, requirement). For API compatibility.
        author: Optional author. For API compatibility.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "text": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate assumption_type (for API compatibility)
    if assumption_type not in ASSUMPTION_TYPES:
        return (
            None,
            f"Invalid assumption_type '{assumption_type}'. Must be one of: {', '.join(ASSUMPTION_TYPES)}",
        )

    # Validate text
    if not text or not text.strip():
        return None, "Assumption text is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Ensure metadata.assumptions exists
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "assumptions" not in spec_data["metadata"]:
        spec_data["metadata"]["assumptions"] = []

    assumptions = spec_data["metadata"]["assumptions"]

    # Schema expects strings, so store text directly
    assumption_text = text.strip()

    # Check for duplicates
    if assumption_text in assumptions:
        return None, f"Assumption already exists: {assumption_text[:50]}..."

    # Add to assumptions array (as string per schema)
    assumptions.append(assumption_text)

    # Update last_updated
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    # Return index as "ID" for API compatibility
    assumption_index = len(assumptions)

    return {
        "spec_id": spec_id,
        "assumption_id": f"a-{assumption_index}",
        "text": assumption_text,
        "type": assumption_type,
        "author": author,
        "index": assumption_index,
    }, None


def add_revision(
    spec_id: str,
    version: str,
    changelog: str,
    author: Optional[str] = None,
    modified_by: Optional[str] = None,
    review_triggered_by: Optional[str] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a revision entry to a specification's revision_history array.

    Args:
        spec_id: Specification ID to add revision to.
        version: Version number (e.g., "1.0", "1.1", "2.0").
        changelog: Description of changes made in this revision.
        author: Optional author who made the revision.
        modified_by: Optional tool or command that made the modification.
        review_triggered_by: Optional path to review report that triggered this revision.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "version": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate version
    if not version or not version.strip():
        return None, "Version is required"

    # Validate changelog
    if not changelog or not changelog.strip():
        return None, "Changelog is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Ensure metadata.revision_history exists
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "revision_history" not in spec_data["metadata"]:
        spec_data["metadata"]["revision_history"] = []

    revision_history = spec_data["metadata"]["revision_history"]

    # Create revision entry per schema
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    revision_entry = {
        "version": version.strip(),
        "date": now,
        "changelog": changelog.strip(),
    }

    # Add optional fields if provided
    if author:
        revision_entry["author"] = author.strip()
    if modified_by:
        revision_entry["modified_by"] = modified_by.strip()
    if review_triggered_by:
        revision_entry["review_triggered_by"] = review_triggered_by.strip()

    # Append to revision history
    revision_history.append(revision_entry)

    # Update last_updated
    spec_data["last_updated"] = now

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "version": revision_entry["version"],
        "date": revision_entry["date"],
        "changelog": revision_entry["changelog"],
        "author": author,
        "modified_by": modified_by,
        "review_triggered_by": review_triggered_by,
        "revision_index": len(revision_history),
    }, None


def list_assumptions(
    spec_id: str,
    assumption_type: Optional[str] = None,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    List assumptions from a specification.

    Args:
        spec_id: Specification ID to list assumptions from.
        assumption_type: Optional filter by type (constraint, requirement).
            Note: Since assumptions are stored as strings, this filter is
            provided for API compatibility but has no effect.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "assumptions": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate assumption_type if provided
    if assumption_type and assumption_type not in ASSUMPTION_TYPES:
        return (
            None,
            f"Invalid assumption_type '{assumption_type}'. Must be one of: {', '.join(ASSUMPTION_TYPES)}",
        )

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Get assumptions from metadata
    assumptions = spec_data.get("metadata", {}).get("assumptions", [])

    # Build assumption list with indices
    assumption_list = []
    for i, assumption in enumerate(assumptions, 1):
        if isinstance(assumption, str):
            assumption_list.append(
                {
                    "id": f"a-{i}",
                    "text": assumption,
                    "index": i,
                }
            )

    return {
        "spec_id": spec_id,
        "assumptions": assumption_list,
        "total_count": len(assumption_list),
        "filter_type": assumption_type,
    }, None


# Valid frontmatter keys that can be updated
# Note: assumptions and revision_history have dedicated functions
FRONTMATTER_KEYS = (
    "title",
    "description",
    "mission",
    "objectives",
    "complexity",
    "estimated_hours",
    "owner",
    "status",
    "category",
    "progress_percentage",
    "current_phase",
)


def update_frontmatter(
    spec_id: str,
    key: str,
    value: Any,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update a top-level metadata field in a specification.

    Updates fields in the spec's metadata block. For arrays like assumptions
    or revision_history, use the dedicated add_assumption() and add_revision()
    functions instead.

    Args:
        spec_id: Specification ID to update.
        key: Metadata key to update (e.g., "title", "status", "description").
        value: New value for the key.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "key": ..., "value": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate key
    if not key or not key.strip():
        return None, "Key is required"

    key = key.strip()

    # Block array fields that have dedicated functions
    if key in ("assumptions", "revision_history"):
        return (
            None,
            f"Use dedicated function for '{key}' (add_assumption or add_revision)",
        )

    # Validate value is not None (but allow empty string, 0, False, etc.)
    if value is None:
        return None, "Value cannot be None"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set SDD_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Ensure metadata exists
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}

    # Get previous value for result (check appropriate location)
    if key in ("status", "progress_percentage", "current_phase"):
        previous_value = spec_data.get(key)
    else:
        previous_value = spec_data["metadata"].get(key)

    # Process value based on type
    if isinstance(value, str):
        value = value.strip() if value else value

    # Computed fields (status, progress_percentage, current_phase) are now
    # stored only at top-level. Title is kept in metadata for descriptive purposes.
    if key in ("status", "progress_percentage", "current_phase"):
        # Update top-level only (canonical location for computed fields)
        spec_data[key] = value
    else:
        # Regular metadata field
        spec_data["metadata"][key] = value
        # Also sync title to top-level if updating it
        if key == "title":
            spec_data[key] = value

    # Update last_updated
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "key": key,
        "value": value,
        "previous_value": previous_value,
    }, None


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
                        warnings.append(
                            f"max_replacements limit ({max_replacements}) reached"
                        )
                        limit_reached = True
                    else:
                        total_replacements += count
                        nodes_affected.add(node_id)
                        changes.append({
                            "node_id": node_id,
                            "field": "title",
                            "old": title,
                            "new": new_title,
                            "replacement_count": count,
                        })
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
                            warnings.append(
                                f"max_replacements limit ({max_replacements}) reached"
                            )
                            limit_reached = True
                        else:
                            total_replacements += count
                            nodes_affected.add(node_id)
                            changes.append({
                                "node_id": node_id,
                                "field": "description",
                                "old": description,
                                "new": new_description,
                                "replacement_count": count,
                            })
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


# Completeness check constants
_CC_WEIGHT_TITLES = 0.20
_CC_WEIGHT_DESCRIPTIONS = 0.30
_CC_WEIGHT_FILE_PATHS = 0.25
_CC_WEIGHT_ESTIMATES = 0.25


def check_spec_completeness(
    spec_id: str,
    *,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Check spec completeness and calculate a score (0-100).

    Evaluates spec quality by checking for:
    - Empty titles
    - Missing task descriptions
    - Missing file_path for implementation/refactoring tasks
    - Missing estimated_hours

    Args:
        spec_id: Specification ID to check.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "completeness_score": ..., ...}, None)
        On failure: (None, "error message")
    """
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
            "completeness_score": 100,
            "categories": {},
            "issues": [],
            "message": "No hierarchy nodes to check",
        }, None

    # Helper functions
    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _has_description(metadata: Dict[str, Any]) -> bool:
        if _nonempty_string(metadata.get("description")):
            return True
        details = metadata.get("details")
        if _nonempty_string(details):
            return True
        if isinstance(details, list):
            return any(_nonempty_string(item) for item in details)
        return False

    # Tracking
    issues: List[Dict[str, Any]] = []
    categories: Dict[str, Dict[str, Any]] = {
        "titles": {"complete": 0, "total": 0, "score": 0.0},
        "descriptions": {"complete": 0, "total": 0, "score": 0.0},
        "file_paths": {"complete": 0, "total": 0, "score": 0.0},
        "estimates": {"complete": 0, "total": 0, "score": 0.0},
    }

    # Check each node
    for node_id, node in hierarchy.items():
        if node_id == "spec-root":
            continue
        if not isinstance(node, dict):
            continue

        node_type = node.get("type", "")
        title = node.get("title", "")
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Check title (all nodes)
        categories["titles"]["total"] += 1
        if _nonempty_string(title):
            categories["titles"]["complete"] += 1
        else:
            issues.append({
                "node_id": node_id,
                "category": "titles",
                "message": "Empty or missing title",
            })

        # Check description (tasks and verify nodes only)
        if node_type in ("task", "verify"):
            categories["descriptions"]["total"] += 1
            if _has_description(metadata):
                categories["descriptions"]["complete"] += 1
            else:
                issues.append({
                    "node_id": node_id,
                    "category": "descriptions",
                    "message": "Missing description",
                })

            # Check file_path (implementation/refactoring tasks only)
            task_category = metadata.get("task_category", "")
            if task_category in ("implementation", "refactoring"):
                categories["file_paths"]["total"] += 1
                if _nonempty_string(metadata.get("file_path")):
                    categories["file_paths"]["complete"] += 1
                else:
                    issues.append({
                        "node_id": node_id,
                        "category": "file_paths",
                        "message": "Missing file_path for implementation task",
                    })

            # Check estimated_hours (tasks only)
            if node_type == "task":
                categories["estimates"]["total"] += 1
                est = metadata.get("estimated_hours")
                if isinstance(est, (int, float)) and est > 0:
                    categories["estimates"]["complete"] += 1
                else:
                    issues.append({
                        "node_id": node_id,
                        "category": "estimates",
                        "message": "Missing or invalid estimated_hours",
                    })

    # Calculate category scores
    for cat_data in categories.values():
        if cat_data["total"] > 0:
            cat_data["score"] = round(cat_data["complete"] / cat_data["total"], 2)
        else:
            cat_data["score"] = 1.0  # No items to check = complete

    # Calculate weighted completeness score
    weighted_score = 0.0
    total_weight = 0.0

    if categories["titles"]["total"] > 0:
        weighted_score += categories["titles"]["score"] * _CC_WEIGHT_TITLES
        total_weight += _CC_WEIGHT_TITLES

    if categories["descriptions"]["total"] > 0:
        weighted_score += categories["descriptions"]["score"] * _CC_WEIGHT_DESCRIPTIONS
        total_weight += _CC_WEIGHT_DESCRIPTIONS

    if categories["file_paths"]["total"] > 0:
        weighted_score += categories["file_paths"]["score"] * _CC_WEIGHT_FILE_PATHS
        total_weight += _CC_WEIGHT_FILE_PATHS

    if categories["estimates"]["total"] > 0:
        weighted_score += categories["estimates"]["score"] * _CC_WEIGHT_ESTIMATES
        total_weight += _CC_WEIGHT_ESTIMATES

    # Normalize score
    if total_weight > 0:
        completeness_score = int(round((weighted_score / total_weight) * 100))
    else:
        completeness_score = 100  # Nothing to check

    return {
        "spec_id": spec_id,
        "completeness_score": completeness_score,
        "categories": categories,
        "issues": issues,
        "issue_count": len(issues),
    }, None


# Duplicate detection constants
_DD_DEFAULT_THRESHOLD = 0.8
_DD_MAX_PAIRS = 100
_DD_VALID_SCOPES = {"titles", "descriptions", "both"}


def detect_duplicate_tasks(
    spec_id: str,
    *,
    scope: str = "titles",
    threshold: float = _DD_DEFAULT_THRESHOLD,
    max_pairs: int = _DD_MAX_PAIRS,
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Detect duplicate or near-duplicate tasks in a spec.

    Uses text similarity to find tasks with similar titles or descriptions.

    Args:
        spec_id: Specification ID to check.
        scope: What to compare - "titles", "descriptions", or "both".
        threshold: Similarity threshold (0.0-1.0). Default 0.8.
        max_pairs: Maximum duplicate pairs to return. Default 100.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "duplicates": [...], ...}, None)
        On failure: (None, "error message")
    """
    from difflib import SequenceMatcher

    # Validate scope
    if scope not in _DD_VALID_SCOPES:
        return None, f"scope must be one of: {sorted(_DD_VALID_SCOPES)}"

    # Validate threshold
    if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        return None, "threshold must be a number between 0.0 and 1.0"

    # Validate max_pairs
    if not isinstance(max_pairs, int) or max_pairs <= 0:
        return None, "max_pairs must be a positive integer"

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
            "duplicates": [],
            "duplicate_count": 0,
            "scope": scope,
            "threshold": threshold,
            "message": "No hierarchy nodes to check",
        }, None

    # Collect tasks/verify nodes with their text
    nodes: List[Dict[str, Any]] = []
    for node_id, node in hierarchy.items():
        if node_id == "spec-root":
            continue
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "")
        if node_type not in ("task", "verify"):
            continue

        title = node.get("title", "") or ""
        metadata = node.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        description = metadata.get("description", "") or ""

        nodes.append({
            "id": node_id,
            "title": title.strip().lower(),
            "description": description.strip().lower(),
        })

    # Compare pairs
    duplicates: List[Dict[str, Any]] = []
    truncated = False
    total_compared = 0

    def similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    for i, node_a in enumerate(nodes):
        if len(duplicates) >= max_pairs:
            truncated = True
            break
        for node_b in nodes[i + 1:]:
            total_compared += 1
            if len(duplicates) >= max_pairs:
                truncated = True
                break

            # Calculate similarity based on scope
            if scope == "titles":
                sim = similarity(node_a["title"], node_b["title"])
            elif scope == "descriptions":
                sim = similarity(node_a["description"], node_b["description"])
            else:  # both
                title_sim = similarity(node_a["title"], node_b["title"])
                desc_sim = similarity(node_a["description"], node_b["description"])
                sim = max(title_sim, desc_sim)

            if sim >= threshold:
                duplicates.append({
                    "node_a": node_a["id"],
                    "node_b": node_b["id"],
                    "similarity": round(sim, 2),
                    "scope": scope,
                })

    result: Dict[str, Any] = {
        "spec_id": spec_id,
        "duplicates": duplicates,
        "duplicate_count": len(duplicates),
        "scope": scope,
        "threshold": threshold,
        "nodes_checked": len(nodes),
        "pairs_compared": total_compared,
    }

    if truncated:
        result["truncated"] = True
        result["warnings"] = [f"Results limited to {max_pairs} pairs"]

    return result, None
