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
from typing import Optional, Dict, Any, List, Tuple

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

    requires_rich_tasks = _requires_rich_task_fields(spec_data)

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
        if _nonempty_string(description):
            return "description", description.strip()
        details = task_def.get("details")
        if _nonempty_string(details):
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
    if error:
        return None, error

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
