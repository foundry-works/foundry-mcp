"""
Spec I/O operations: discovery, loading, saving, backups, and listing.

All file-system interactions for specs live here.  The only intra-package
dependency is ``_constants``; every other spec sub-module may import from
this module freely (via the package ``__init__``).
"""

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from foundry_mcp.core.spec._constants import (
    DEFAULT_BACKUP_PAGE_SIZE,
    DEFAULT_DIFF_MAX_RESULTS,
    DEFAULT_MAX_BACKUPS,
    MAX_BACKUP_PAGE_SIZE,
)


# ---------------------------------------------------------------------------
# Git / directory discovery
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Spec field migration (backward compat)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


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
            "failed",
        ]:
            return False

    return True


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


# ---------------------------------------------------------------------------
# Backup / retention
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Diff / rollback
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# List / generate
# ---------------------------------------------------------------------------


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
