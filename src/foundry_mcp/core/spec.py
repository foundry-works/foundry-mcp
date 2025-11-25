"""
JSON spec file operations for SDD workflows.
Provides loading, saving, finding, and listing specs with atomic writes and backups.
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List


def find_git_root() -> Optional[Path]:
    """Find the root of the git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True
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
        return ((p / "pending").is_dir() or
                (p / "active").is_dir() or
                (p / "completed").is_dir() or
                (p / "archived").is_dir())

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


def resolve_spec_file(spec_name_or_path: str, specs_dir: Optional[Path] = None) -> Optional[Path]:
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
        if spec_file.exists() and spec_file.suffix == '.json':
            return spec_file
        return None

    search_name = spec_name_or_path
    if spec_name_or_path.endswith('.json'):
        spec_file = path.resolve()
        if spec_file.exists() and spec_file.suffix == '.json':
            return spec_file
        search_name = path.stem

    if specs_dir is None:
        specs_dir = find_specs_directory()

    if not specs_dir:
        return None

    return find_spec_file(search_name, specs_dir)


def load_spec(spec_id: str, specs_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
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
        with open(spec_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_spec(
    spec_id: str,
    spec_data: Dict[str, Any],
    specs_dir: Optional[Path] = None,
    backup: bool = True,
    validate: bool = True
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

    spec_data["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    if backup:
        backup_spec(spec_id, specs_dir)

    temp_file = spec_file.with_suffix('.tmp')
    try:
        with open(temp_file, 'w') as f:
            json.dump(spec_data, f, indent=2)
        temp_file.replace(spec_file)
        return True
    except (IOError, OSError):
        if temp_file.exists():
            temp_file.unlink()
        return False


def backup_spec(spec_id: str, specs_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create a backup copy of the JSON spec file in the .backups/ directory.

    Args:
        spec_id: Specification ID or path to spec file
        specs_dir: Path to specs directory (optional, auto-detected if not provided)

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

    backups_dir = specs_dir / ".backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    backup_file = backups_dir / f"{spec_id}.backup"

    try:
        shutil.copy2(spec_file, backup_file)
        return backup_file
    except (IOError, OSError):
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
        if node_data["status"] not in ["pending", "in_progress", "completed", "blocked"]:
            return False

    return True


def list_specs(
    specs_dir: Optional[Path] = None,
    status: Optional[str] = None
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
                1 for task in hierarchy.values()
                if task.get("status") == "completed"
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
    specs_info.sort(key=lambda s: (0 if s.get("status") == "active" else 1, -s.get("progress_percentage", 0)))

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


def update_node(spec_data: Dict[str, Any], node_id: str, updates: Dict[str, Any]) -> bool:
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
