"""Spec structure hashing for integrity validation.

Provides deterministic hashing of spec structure for fast equality checking,
file metadata retrieval for optimization, and structural diff computation
for human-readable reporting during rebase and spec drift detection.

Key functions:
- compute_spec_structure_hash(): SHA-256 of structural elements
- get_spec_file_metadata(): mtime and file size for quick checks
- compute_structural_diff(): Human-readable comparison of two structures
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpecFileMetadata:
    """Metadata for a spec file."""

    mtime: float
    file_size: int


@dataclass
class StructuralDiff:
    """Result of comparing two spec structures.

    Contains lists of added/removed phases and tasks for human-readable
    reporting during rebase operations and spec drift detection.
    """

    added_phases: List[str] = field(default_factory=list)
    removed_phases: List[str] = field(default_factory=list)
    added_tasks: List[str] = field(default_factory=list)
    removed_tasks: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any structural changes."""
        return bool(
            self.added_phases
            or self.removed_phases
            or self.added_tasks
            or self.removed_tasks
        )

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary for serialization."""
        return {
            "added_phases": self.added_phases,
            "removed_phases": self.removed_phases,
            "added_tasks": self.added_tasks,
            "removed_tasks": self.removed_tasks,
        }


def compute_spec_structure_hash(spec_data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of spec structural elements.

    Creates a deterministic hash based only on structural elements:
    - Phase IDs (sorted)
    - Task IDs (sorted)
    - Task-to-phase parent mappings
    - Phase ordering (sequence_index)

    Non-structural elements (descriptions, metadata, titles) are excluded
    to tolerate cosmetic changes without triggering mismatch.

    Args:
        spec_data: Parsed spec JSON as dictionary

    Returns:
        SHA-256 hash hex string (64 characters)

    Example:
        >>> spec = {"phases": [{"id": "phase-1", "tasks": [{"id": "task-1"}]}]}
        >>> compute_spec_structure_hash(spec)
        'a1b2c3...'
    """
    # Extract structural elements in deterministic order
    structure: Dict[str, Any] = {}

    # Get phases array (handle both 'phases' key and direct array)
    phases = spec_data.get("phases", [])
    if not isinstance(phases, list):
        phases = []

    # Extract phase IDs and their ordering
    phase_ids: List[str] = []
    phase_ordering: List[tuple] = []  # (phase_id, sequence_index)

    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_id = phase.get("id", "")
        if phase_id:
            phase_ids.append(phase_id)
            seq_idx = phase.get("sequence_index", 0)
            phase_ordering.append((phase_id, seq_idx))

    # Sort phase IDs for deterministic ordering
    structure["phase_ids"] = sorted(phase_ids)

    # Sort phase ordering by phase_id for determinism
    structure["phase_ordering"] = sorted(phase_ordering, key=lambda x: x[0])

    # Extract task IDs and task-to-phase mappings
    task_ids: List[str] = []
    task_phase_mappings: List[tuple] = []  # (task_id, parent_phase_id)

    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_id = phase.get("id", "")
        tasks = phase.get("tasks", [])
        if not isinstance(tasks, list):
            continue

        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = task.get("id", "")
            if task_id:
                task_ids.append(task_id)
                task_phase_mappings.append((task_id, phase_id))

    # Sort for determinism
    structure["task_ids"] = sorted(task_ids)
    structure["task_phase_mappings"] = sorted(
        task_phase_mappings, key=lambda x: (x[0], x[1])
    )

    # Serialize with sorted keys for determinism
    canonical_json = json.dumps(
        structure,
        separators=(",", ":"),
        sort_keys=True,
    )

    # Compute SHA-256 hash
    hash_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    logger.debug(
        "Computed spec structure hash: %s (phases=%d, tasks=%d)",
        hash_digest[:16] + "...",
        len(phase_ids),
        len(task_ids),
    )

    return hash_digest


def get_spec_file_metadata(spec_path: Path) -> Optional[SpecFileMetadata]:
    """Get file metadata for a spec file.

    Retrieves mtime and file size for quick staleness checks without
    parsing the entire file. Used as an optimization before re-hashing.

    Args:
        spec_path: Path to the spec JSON file

    Returns:
        SpecFileMetadata with mtime and file_size, or None if file not found

    Example:
        >>> metadata = get_spec_file_metadata(Path("specs/my-spec.json"))
        >>> if metadata:
        ...     print(f"mtime={metadata.mtime}, size={metadata.file_size}")
    """
    if not spec_path.exists():
        logger.debug("Spec file not found: %s", spec_path)
        return None

    try:
        stat = spec_path.stat()
        return SpecFileMetadata(
            mtime=stat.st_mtime,
            file_size=stat.st_size,
        )
    except OSError as exc:
        logger.warning("Failed to get metadata for %s: %s", spec_path, exc)
        return None


def compute_structural_diff(
    old_structure: Dict[str, Any],
    new_structure: Dict[str, Any],
) -> StructuralDiff:
    """Compute human-readable diff between two spec structures.

    Compares phase and task membership between two full spec structures
    (not hashes) to identify added and removed elements. Used for
    rebase reporting and spec drift detection.

    This function operates on full structures to provide meaningful
    human-readable output. Use compute_spec_structure_hash() for
    fast equality checks.

    Args:
        old_structure: Previous spec structure (parsed JSON)
        new_structure: Current spec structure (parsed JSON)

    Returns:
        StructuralDiff with lists of added/removed phases and tasks

    Example:
        >>> old = {"phases": [{"id": "phase-1", "tasks": [{"id": "task-1"}]}]}
        >>> new = {"phases": [{"id": "phase-1", "tasks": [{"id": "task-1"}, {"id": "task-2"}]}]}
        >>> diff = compute_structural_diff(old, new)
        >>> diff.added_tasks
        ['task-2']
    """
    # Extract phase sets
    old_phases = _extract_phase_ids(old_structure)
    new_phases = _extract_phase_ids(new_structure)

    # Extract task sets
    old_tasks = _extract_task_ids(old_structure)
    new_tasks = _extract_task_ids(new_structure)

    # Compute differences
    added_phases = sorted(new_phases - old_phases)
    removed_phases = sorted(old_phases - new_phases)
    added_tasks = sorted(new_tasks - old_tasks)
    removed_tasks = sorted(old_tasks - new_tasks)

    diff = StructuralDiff(
        added_phases=added_phases,
        removed_phases=removed_phases,
        added_tasks=added_tasks,
        removed_tasks=removed_tasks,
    )

    if diff.has_changes:
        logger.info(
            "Structural diff: +%d/-%d phases, +%d/-%d tasks",
            len(added_phases),
            len(removed_phases),
            len(added_tasks),
            len(removed_tasks),
        )
    else:
        logger.debug("No structural changes detected")

    return diff


def _extract_phase_ids(spec_data: Dict[str, Any]) -> set:
    """Extract set of phase IDs from spec structure.

    Args:
        spec_data: Parsed spec JSON

    Returns:
        Set of phase ID strings
    """
    phases = spec_data.get("phases", [])
    if not isinstance(phases, list):
        return set()

    return {
        phase.get("id")
        for phase in phases
        if isinstance(phase, dict) and phase.get("id")
    }


def _extract_task_ids(spec_data: Dict[str, Any]) -> set:
    """Extract set of task IDs from spec structure.

    Args:
        spec_data: Parsed spec JSON

    Returns:
        Set of task ID strings
    """
    phases = spec_data.get("phases", [])
    if not isinstance(phases, list):
        return set()

    task_ids = set()
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        tasks = phase.get("tasks", [])
        if not isinstance(tasks, list):
            continue

        for task in tasks:
            if isinstance(task, dict) and task.get("id"):
                task_ids.add(task.get("id"))

    return task_ids
