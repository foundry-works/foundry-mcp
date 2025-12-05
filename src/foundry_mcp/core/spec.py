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
TEMPLATES = ("simple", "medium", "complex", "security")
CATEGORIES = ("investigation", "implementation", "refactoring", "decision", "research")

# Valid verification types for verify nodes
# - run-tests: Automated tests via mcp__foundry-mcp__test-run
# - fidelity: Implementation-vs-spec comparison via mcp__foundry-mcp__spec-review-fidelity
VERIFICATION_TYPES = ("run-tests", "fidelity")


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
            return json.load(f)
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
    pattern = re.compile(r"^phase-(\\d+)$")
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


def get_template_structure(template: str, category: str) -> Dict[str, Any]:
    """
    Get the hierarchical structure for a spec template.

    All templates include per-phase verification (auto + fidelity) for each phase.

    Args:
        template: Template type (simple, medium, complex, security).
        category: Default task category.

    Returns:
        Hierarchy dict for the spec.
    """
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
                "details": "Document the requirements and acceptance criteria",
                "category": category,
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
                "details": "Implement the main features",
                "category": category,
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
                "details": "Review for security vulnerabilities",
                "category": "investigation",
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


def create_spec(
    name: str,
    template: str = "medium",
    category: str = "implementation",
    specs_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Create a new specification file from a template.

    Args:
        name: Human-readable name for the specification.
        template: Template type (simple, medium, complex, security). Default: medium.
        category: Default task category. Default: implementation.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "spec_path": ..., ...}, None)
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

    # Generate spec ID
    spec_id = generate_spec_id(name)

    # Check if spec already exists
    spec_path = pending_dir / f"{spec_id}.json"
    if spec_path.exists():
        return None, f"Specification already exists: {spec_id}"

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

    spec_data = {
        "spec_id": spec_id,
        "title": name,
        "generated": now,
        "last_updated": now,
        "metadata": {
            "description": "",
            "objectives": [],
            "complexity": "medium" if template in ("medium", "complex") else "low",
            "estimated_hours": estimated_hours,
            "assumptions": [],
            "status": "pending",
            "owner": "",
            "progress_percentage": 0,
            "current_phase": "phase-1",
            "category": category,
            "template": template,
        },
        "progress_percentage": 0,
        "status": "pending",
        "current_phase": "phase-1",
        "hierarchy": hierarchy,
        "journal": [],
    }

    # Write the spec file
    try:
        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=2)
    except (IOError, OSError) as e:
        return None, f"Failed to write spec file: {e}"

    # Count tasks and phases
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
        elif isinstance(assumption, dict):
            # Handle legacy object format
            assumption_list.append(
                {
                    "id": assumption.get("id", f"a-{i}"),
                    "text": assumption.get("text", str(assumption)),
                    "type": assumption.get("type"),
                    "author": assumption.get("author"),
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

    # Get previous value for result
    previous_value = spec_data["metadata"].get(key)

    # Process value based on type
    if isinstance(value, str):
        value = value.strip() if value else value

    # Update the metadata field
    spec_data["metadata"][key] = value

    # Also update top-level fields if they exist (for backward compatibility)
    # Some fields like title, status, progress_percentage exist at both levels
    if key in ("title", "status", "progress_percentage", "current_phase"):
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
