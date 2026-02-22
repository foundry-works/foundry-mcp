"""
Lifecycle operations for SDD spec files.
Provides spec status transitions: move, activate, complete, archive.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Data structures


@dataclass
class MoveResult:
    """
    Result of moving a spec between status folders.
    """

    success: bool
    spec_id: str
    from_folder: str
    to_folder: str
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class LifecycleState:
    """
    Current lifecycle state of a spec.
    """

    spec_id: str
    folder: str  # pending, active, completed, archived
    status: str  # from spec-root
    progress_percentage: float
    total_tasks: int
    completed_tasks: int
    can_complete: bool
    can_archive: bool


# Constants

VALID_FOLDERS = {"pending", "active", "completed", "archived"}
FOLDER_SEARCH_ORDER = ("pending", "active", "completed", "archived")

FOLDER_TRANSITIONS = {
    "pending": ["active", "archived"],
    "active": ["pending", "completed", "archived"],
    "completed": ["active", "archived"],
    "archived": ["pending", "active"],
}


# Main lifecycle functions


def move_spec(
    spec_id: str,
    to_folder: str,
    specs_dir: Path,
    update_status: bool = True,
) -> MoveResult:
    """
    Move a spec between status folders.

    Args:
        spec_id: Specification ID
        to_folder: Target folder (pending, active, completed, archived)
        specs_dir: Path to specs directory
        update_status: Whether to update spec-root status to match folder

    Returns:
        MoveResult with operation outcome
    """
    if to_folder not in VALID_FOLDERS:
        return MoveResult(
            success=False,
            spec_id=spec_id,
            from_folder="",
            to_folder=to_folder,
            error=f"Invalid folder: {to_folder}. Must be one of: {VALID_FOLDERS}",
        )

    # Find current location
    current_folder, current_path = _find_spec_location(spec_id, specs_dir)

    if not current_path or not current_folder:
        return MoveResult(
            success=False,
            spec_id=spec_id,
            from_folder="",
            to_folder=to_folder,
            error=f"Spec not found: {spec_id}",
        )

    if current_folder == to_folder:
        return MoveResult(
            success=True,
            spec_id=spec_id,
            from_folder=current_folder,
            to_folder=to_folder,
            old_path=str(current_path),
            new_path=str(current_path),
        )

    # Validate transition
    allowed = FOLDER_TRANSITIONS.get(current_folder, [])
    if to_folder not in allowed:
        return MoveResult(
            success=False,
            spec_id=spec_id,
            from_folder=current_folder,
            to_folder=to_folder,
            error=f"Cannot move from {current_folder} to {to_folder}. Allowed: {allowed}",
        )

    # Ensure target folder exists
    target_dir = specs_dir / to_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    # Calculate new path
    new_path = target_dir / current_path.name

    # Check for conflicts
    if new_path.exists():
        return MoveResult(
            success=False,
            spec_id=spec_id,
            from_folder=current_folder,
            to_folder=to_folder,
            error=f"Target already exists: {new_path}",
        )

    try:
        # Move the file
        shutil.move(str(current_path), str(new_path))

        # Update spec status if requested
        if update_status:
            _update_spec_folder_status(new_path, to_folder)

        return MoveResult(
            success=True,
            spec_id=spec_id,
            from_folder=current_folder,
            to_folder=to_folder,
            old_path=str(current_path),
            new_path=str(new_path),
        )

    except OSError as e:
        return MoveResult(
            success=False,
            spec_id=spec_id,
            from_folder=current_folder,
            to_folder=to_folder,
            error=f"Failed to move file: {e}",
        )


def activate_spec(spec_id: str, specs_dir: Path) -> MoveResult:
    """
    Activate a spec (move from pending to active).

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory

    Returns:
        MoveResult with operation outcome
    """
    return move_spec(spec_id, "active", specs_dir)


def complete_spec(
    spec_id: str,
    specs_dir: Path,
    force: bool = False,
) -> MoveResult:
    """
    Mark a spec as completed (move to completed folder).

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory
        force: Force completion even if tasks are incomplete

    Returns:
        MoveResult with operation outcome
    """
    # Check if spec can be completed
    if not force:
        state = get_lifecycle_state(spec_id, specs_dir)
        if state and not state.can_complete:
            return MoveResult(
                success=False,
                spec_id=spec_id,
                from_folder=state.folder,
                to_folder="completed",
                error=f"Cannot complete spec: {state.completed_tasks}/{state.total_tasks} tasks done ({state.progress_percentage:.0f}%)",
            )

    return move_spec(spec_id, "completed", specs_dir)


def archive_spec(spec_id: str, specs_dir: Path) -> MoveResult:
    """
    Archive a spec (move to archived folder).

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory

    Returns:
        MoveResult with operation outcome
    """
    return move_spec(spec_id, "archived", specs_dir)


def get_lifecycle_state(spec_id: str, specs_dir: Path) -> Optional[LifecycleState]:
    """
    Get the current lifecycle state of a spec.

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory

    Returns:
        LifecycleState or None if spec not found
    """
    folder, path = _find_spec_location(spec_id, specs_dir)

    if not path or not folder:
        return None

    try:
        with open(path, "r") as f:
            spec_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    hierarchy = spec_data.get("hierarchy", {})
    root = hierarchy.get("spec-root", {})

    total_tasks = root.get("total_tasks", 0)
    completed_tasks = root.get("completed_tasks", 0)
    status = root.get("status", "pending")

    progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    # Can complete if all tasks done or in active folder with 100% progress
    can_complete = progress >= 100 or status == "completed"

    # Can archive from any folder
    can_archive = True

    return LifecycleState(
        spec_id=spec_id,
        folder=folder,
        status=status,
        progress_percentage=progress,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        can_complete=can_complete,
        can_archive=can_archive,
    )


def list_specs_by_folder(
    specs_dir: Path,
    folder: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    List specs organized by folder.

    Args:
        specs_dir: Path to specs directory
        folder: Optional filter to specific folder

    Returns:
        Dict mapping folder names to lists of spec summaries
    """
    result: Dict[str, List[Dict[str, Any]]] = {}

    folders = [folder] if folder else list(VALID_FOLDERS)

    for f in folders:
        folder_path = specs_dir / f
        if not folder_path.exists():
            result[f] = []
            continue

        specs = []
        for spec_file in folder_path.glob("*.json"):
            try:
                with open(spec_file, "r") as file:
                    data = json.load(file)

                hierarchy = data.get("hierarchy", {})
                root = hierarchy.get("spec-root", {})
                metadata = data.get("metadata", {})

                total = root.get("total_tasks", 0)
                completed = root.get("completed_tasks", 0)

                specs.append(
                    {
                        "spec_id": data.get("spec_id", spec_file.stem),
                        "title": metadata.get("title") or root.get("title", "Untitled"),
                        "status": root.get("status", "pending"),
                        "total_tasks": total,
                        "completed_tasks": completed,
                        "progress": (completed / total * 100) if total > 0 else 0,
                        "path": str(spec_file),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue

        result[f] = specs

    return result


def get_folder_for_spec(spec_id: str, specs_dir: Path) -> Optional[str]:
    """
    Get the folder where a spec is located.

    Args:
        spec_id: Specification ID
        specs_dir: Path to specs directory

    Returns:
        Folder name or None if not found
    """
    folder, _ = _find_spec_location(spec_id, specs_dir)
    return folder


# Helper functions


def _find_spec_location(
    spec_id: str,
    specs_dir: Path,
) -> tuple[Optional[str], Optional[Path]]:
    """
    Find which folder contains a spec.

    Returns:
        Tuple of (folder_name, path) or (None, None) if not found
    """
    for folder in FOLDER_SEARCH_ORDER:
        folder_path = specs_dir / folder
        if not folder_path.exists():
            continue

        # Check for exact match
        spec_path = folder_path / f"{spec_id}.json"
        if spec_path.exists():
            return folder, spec_path

        # Check for files that contain the spec_id
        for f in folder_path.glob("*.json"):
            try:
                with open(f, "r") as file:
                    data = json.load(file)
                if data.get("spec_id") == spec_id:
                    return folder, f
            except (OSError, json.JSONDecodeError):
                continue

    return None, None


def _update_spec_folder_status(path: Path, folder: str) -> bool:
    """
    Update spec-root status to match folder.

    Args:
        path: Path to spec file
        folder: Folder name (maps to status)

    Returns:
        True if update successful
    """
    folder_to_status = {
        "pending": "pending",
        "active": "in_progress",
        "completed": "completed",
        "archived": "archived",
    }

    new_status = folder_to_status.get(folder, "pending")

    try:
        with open(path, "r") as f:
            data = json.load(f)

        hierarchy = data.get("hierarchy", {})
        if "spec-root" in hierarchy:
            hierarchy["spec-root"]["status"] = new_status

        # Update last_updated
        data["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return True

    except (OSError, json.JSONDecodeError):
        return False
