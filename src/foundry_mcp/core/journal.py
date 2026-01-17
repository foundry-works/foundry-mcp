"""
Journal and blocker operations for SDD spec files.
Provides journal entry management, task blocking, and unblocking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


# Data structures

@dataclass
class JournalEntry:
    """
    A journal entry in the spec file.
    """
    timestamp: str
    entry_type: str  # status_change, deviation, blocker, decision, note
    title: str
    content: str
    author: str = "claude-code"
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockerInfo:
    """
    Information about a task blocker.
    """
    blocked_at: str
    blocker_type: str  # dependency, technical, resource, decision
    description: str
    ticket: Optional[str] = None
    blocked_by_external: bool = False


@dataclass
class ResolvedBlocker:
    """
    Information about a resolved blocker.
    """
    blocked_at: str
    blocker_type: str
    description: str
    resolved_at: str
    resolution: str
    ticket: Optional[str] = None


# Constants

VALID_ENTRY_TYPES = {"status_change", "deviation", "blocker", "decision", "note"}
VALID_BLOCKER_TYPES = {"dependency", "technical", "resource", "decision"}
VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}


# Journal operations

def add_journal_entry(
    spec_data: Dict[str, Any],
    title: str,
    content: str,
    entry_type: str = "note",
    task_id: Optional[str] = None,
    author: str = "claude-code",
    metadata: Optional[Dict[str, Any]] = None,
) -> JournalEntry:
    """
    Add a journal entry to the spec data.

    Args:
        spec_data: Spec data dictionary (modified in place)
        title: Entry title
        content: Entry content
        entry_type: Type of entry (status_change, deviation, blocker, decision, note)
        task_id: Optional associated task ID
        author: Author of the entry
        metadata: Optional additional metadata

    Returns:
        The created JournalEntry
    """
    timestamp = _get_timestamp()

    entry_data = {
        "timestamp": timestamp,
        "entry_type": entry_type,
        "title": title,
        "content": content,
        "author": author,
        "metadata": metadata or {},
    }

    if task_id:
        entry_data["task_id"] = task_id

    # Ensure journal array exists
    if "journal" not in spec_data or not isinstance(spec_data["journal"], list):
        spec_data["journal"] = []

    spec_data["journal"].append(entry_data)

    # Update last_updated timestamp
    spec_data["last_updated"] = timestamp

    # Clear needs_journaling flag if task_id provided
    if task_id:
        _clear_journaling_flag(spec_data, task_id, timestamp)

    return JournalEntry(
        timestamp=timestamp,
        entry_type=entry_type,
        title=title,
        content=content,
        author=author,
        task_id=task_id,
        metadata=metadata or {},
    )


def get_journal_entries(
    spec_data: Dict[str, Any],
    task_id: Optional[str] = None,
    entry_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[JournalEntry]:
    """
    Get journal entries from spec data.

    Args:
        spec_data: Spec data dictionary
        task_id: Optional filter by task ID
        entry_type: Optional filter by entry type
        limit: Optional limit on number of entries

    Returns:
        List of JournalEntry objects (most recent first)
    """
    journal = spec_data.get("journal", []) or []

    # Filter entries
    filtered = []
    for entry in journal:
        if task_id and entry.get("task_id") != task_id:
            continue
        if entry_type and entry.get("entry_type") != entry_type:
            continue
        filtered.append(entry)

    # Sort by timestamp descending (most recent first)
    filtered.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    # Apply limit
    if limit:
        filtered = filtered[:limit]

    # Convert to JournalEntry objects
    return [
        JournalEntry(
            timestamp=e.get("timestamp", ""),
            entry_type=e.get("entry_type", "note"),
            title=e.get("title", ""),
            content=e.get("content", ""),
            author=e.get("author", ""),
            task_id=e.get("task_id"),
            metadata=e.get("metadata", {}),
        )
        for e in filtered
    ]


def bulk_journal(
    spec_data: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> List[JournalEntry]:
    """
    Add multiple journal entries to the spec data in a single operation.

    This is more efficient than calling add_journal_entry multiple times
    as it updates the spec data once after all entries are added.

    Args:
        spec_data: Spec data dictionary (modified in place)
        entries: List of entry dicts, each with keys:
            - title (required): Entry title
            - content (required): Entry content
            - entry_type (optional): Type of entry (default: "note")
            - task_id (optional): Associated task ID
            - author (optional): Entry author (default: "claude-code")
            - metadata (optional): Additional metadata dict

    Returns:
        List of created JournalEntry objects

    Example:
        >>> entries = [
        ...     {"title": "First entry", "content": "Content 1", "task_id": "task-1"},
        ...     {"title": "Second entry", "content": "Content 2", "task_id": "task-2"},
        ... ]
        >>> results = bulk_journal(spec_data, entries)
        >>> print(f"Added {len(results)} entries")
    """
    if not entries:
        return []

    # Ensure journal array exists
    if "journal" not in spec_data or not isinstance(spec_data["journal"], list):
        spec_data["journal"] = []

    timestamp = _get_timestamp()
    created_entries: List[JournalEntry] = []
    tasks_to_clear: List[str] = []

    for entry_data in entries:
        if not isinstance(entry_data, dict):
            continue

        title = entry_data.get("title", "")
        content = entry_data.get("content", "")

        if not title or not content:
            continue

        entry_type = entry_data.get("entry_type", "note")
        task_id = entry_data.get("task_id")
        author = entry_data.get("author", "claude-code")
        metadata = entry_data.get("metadata", {})

        journal_entry = {
            "timestamp": timestamp,
            "entry_type": entry_type,
            "title": title,
            "content": content,
            "author": author,
            "metadata": metadata,
        }

        if task_id:
            journal_entry["task_id"] = task_id
            tasks_to_clear.append(task_id)

        spec_data["journal"].append(journal_entry)

        created_entries.append(JournalEntry(
            timestamp=timestamp,
            entry_type=entry_type,
            title=title,
            content=content,
            author=author,
            task_id=task_id,
            metadata=metadata,
        ))

    # Update last_updated timestamp once
    spec_data["last_updated"] = timestamp

    # Clear needs_journaling flags for all affected tasks
    for task_id in tasks_to_clear:
        _clear_journaling_flag(spec_data, task_id, timestamp)

    return created_entries


def get_latest_journal_entry(
    spec_data: Dict[str, Any],
    task_id: str,
) -> Optional[JournalEntry]:
    """
    Get the most recent journal entry for a task.

    Args:
        spec_data: Spec data dictionary
        task_id: Task ID to get entry for

    Returns:
        JournalEntry or None if no entries found
    """
    entries = get_journal_entries(spec_data, task_id=task_id, limit=1)
    return entries[0] if entries else None


# Blocker operations

def mark_blocked(
    spec_data: Dict[str, Any],
    task_id: str,
    reason: str,
    blocker_type: str = "dependency",
    ticket: Optional[str] = None,
) -> bool:
    """
    Mark a task as blocked.

    Args:
        spec_data: Spec data dictionary (modified in place)
        task_id: Task to mark as blocked
        reason: Description of the blocker
        blocker_type: Type of blocker (dependency, technical, resource, decision)
        ticket: Optional ticket/issue reference

    Returns:
        True if successful, False if task not found
    """
    hierarchy = spec_data.get("hierarchy", {})
    if task_id not in hierarchy:
        return False

    task = hierarchy[task_id]
    timestamp = _get_timestamp()

    # Build blocker info
    blocker_info = {
        "blocked_at": timestamp,
        "blocker_type": blocker_type,
        "blocker_description": reason,
        "blocked_by_external": blocker_type in {"resource", "dependency"},
    }

    if ticket:
        blocker_info["blocker_ticket"] = ticket

    # Update task
    task["status"] = "blocked"
    metadata = task.setdefault("metadata", {})
    metadata.update(blocker_info)

    # Update last_updated
    spec_data["last_updated"] = timestamp

    # Recalculate progress
    _recalculate_counts(spec_data)

    return True


def unblock(
    spec_data: Dict[str, Any],
    task_id: str,
    resolution: Optional[str] = None,
    new_status: str = "pending",
) -> bool:
    """
    Unblock a task and optionally set its new status.

    Args:
        spec_data: Spec data dictionary (modified in place)
        task_id: Task to unblock
        resolution: Optional description of how blocker was resolved
        new_status: Status to set after unblocking (default: pending)

    Returns:
        True if successful, False if task not found or not blocked
    """
    hierarchy = spec_data.get("hierarchy", {})
    if task_id not in hierarchy:
        return False

    task = hierarchy[task_id]
    if task.get("status") != "blocked":
        return False

    timestamp = _get_timestamp()
    metadata = task.get("metadata", {}).copy()

    # Move blocker info to resolved_blockers
    if "blocker_description" in metadata:
        resolved_blockers = metadata.setdefault("resolved_blockers", [])
        resolved_blockers.append({
            "blocked_at": metadata.get("blocked_at"),
            "blocker_type": metadata.get("blocker_type"),
            "description": metadata.get("blocker_description"),
            "ticket": metadata.get("blocker_ticket"),
            "resolved_at": timestamp,
            "resolution": resolution or "Blocker resolved",
        })

        # Remove active blocker fields
        for key in ["blocked_at", "blocker_type", "blocker_description", "blocker_ticket", "blocked_by_external"]:
            metadata.pop(key, None)

    # Update task
    task["status"] = new_status
    task["metadata"] = metadata

    # Update last_updated
    spec_data["last_updated"] = timestamp

    # Recalculate progress
    _recalculate_counts(spec_data)

    return True


def get_blocker_info(
    spec_data: Dict[str, Any],
    task_id: str,
) -> Optional[BlockerInfo]:
    """
    Get blocker information for a task.

    Args:
        spec_data: Spec data dictionary
        task_id: Task ID to check

    Returns:
        BlockerInfo if task is blocked, None otherwise
    """
    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)

    if not task or task.get("status") != "blocked":
        return None

    metadata = task.get("metadata", {})
    if "blocker_description" not in metadata:
        return None

    return BlockerInfo(
        blocked_at=metadata.get("blocked_at", ""),
        blocker_type=metadata.get("blocker_type", ""),
        description=metadata.get("blocker_description", ""),
        ticket=metadata.get("blocker_ticket"),
        blocked_by_external=metadata.get("blocked_by_external", False),
    )


def get_resolved_blockers(
    spec_data: Dict[str, Any],
    task_id: str,
) -> List[ResolvedBlocker]:
    """
    Get history of resolved blockers for a task.

    Args:
        spec_data: Spec data dictionary
        task_id: Task ID to check

    Returns:
        List of ResolvedBlocker objects
    """
    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)

    if not task:
        return []

    metadata = task.get("metadata", {})
    resolved = metadata.get("resolved_blockers", [])

    return [
        ResolvedBlocker(
            blocked_at=b.get("blocked_at", ""),
            blocker_type=b.get("blocker_type", ""),
            description=b.get("description", ""),
            resolved_at=b.get("resolved_at", ""),
            resolution=b.get("resolution", ""),
            ticket=b.get("ticket"),
        )
        for b in resolved
    ]


def list_blocked_tasks(spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    List all blocked tasks in the spec.

    Args:
        spec_data: Spec data dictionary

    Returns:
        List of dicts with task_id, title, and blocker info
    """
    hierarchy = spec_data.get("hierarchy", {})
    blocked = []

    for node_id, node in hierarchy.items():
        if node.get("status") == "blocked":
            metadata = node.get("metadata", {})
            blocked.append({
                "task_id": node_id,
                "title": node.get("title", ""),
                "blocker_type": metadata.get("blocker_type", "unknown"),
                "blocker_description": metadata.get("blocker_description", ""),
                "blocked_at": metadata.get("blocked_at", ""),
                "ticket": metadata.get("blocker_ticket"),
            })

    return blocked


# Status update with journaling

def update_task_status(
    spec_data: Dict[str, Any],
    task_id: str,
    new_status: str,
    note: Optional[str] = None,
) -> bool:
    """
    Update a task's status with automatic progress recalculation.

    Args:
        spec_data: Spec data dictionary (modified in place)
        task_id: Task to update
        new_status: New status (pending, in_progress, completed, blocked)
        note: Optional note about the status change

    Returns:
        True if successful, False if task not found or invalid status
    """
    if new_status not in VALID_STATUSES:
        return False

    hierarchy = spec_data.get("hierarchy", {})
    if task_id not in hierarchy:
        return False

    task = hierarchy[task_id]
    timestamp = _get_timestamp()

    # Update status
    task["status"] = new_status

    # Update metadata
    metadata = task.setdefault("metadata", {})

    if new_status == "in_progress":
        metadata["started_at"] = timestamp
    elif new_status == "completed":
        metadata["completed_at"] = timestamp
        metadata["needs_journaling"] = True

        # Auto-calculate actual_hours if started_at exists and not manually set
        if "started_at" in metadata and "actual_hours" not in metadata:
            started_at = datetime.fromisoformat(metadata["started_at"].replace("Z", "+00:00"))
            completed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            metadata["actual_hours"] = round((completed_at - started_at).total_seconds() / 3600, 2)

    if note:
        metadata["status_note"] = note

    # Update last_updated
    spec_data["last_updated"] = timestamp

    # Recalculate progress
    _recalculate_counts(spec_data)

    return True


def mark_task_journaled(
    spec_data: Dict[str, Any],
    task_id: str,
) -> bool:
    """
    Mark a task as journaled (clear needs_journaling flag).

    Args:
        spec_data: Spec data dictionary (modified in place)
        task_id: Task to mark as journaled

    Returns:
        True if successful, False if task not found
    """
    hierarchy = spec_data.get("hierarchy", {})
    if task_id not in hierarchy:
        return False

    task = hierarchy[task_id]
    metadata = task.setdefault("metadata", {})

    if "needs_journaling" in metadata:
        metadata["needs_journaling"] = False
        metadata["journaled_at"] = _get_timestamp()

    return True


def find_unjournaled_tasks(spec_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Find all completed tasks that need journaling.

    Args:
        spec_data: Spec data dictionary

    Returns:
        List of dicts with task_id and title
    """
    hierarchy = spec_data.get("hierarchy", {})
    unjournaled = []

    for node_id, node in hierarchy.items():
        if node.get("status") == "completed":
            metadata = node.get("metadata", {})
            if metadata.get("needs_journaling", False):
                unjournaled.append({
                    "task_id": node_id,
                    "title": node.get("title", ""),
                    "completed_at": metadata.get("completed_at", ""),
                })

    return unjournaled


# Utility functions

def save_journal(
    spec_data: Dict[str, Any],
    spec_path: str,
    create_backup: bool = True,
) -> bool:
    """
    Save spec data with journal to disk.

    Args:
        spec_data: Spec data dictionary
        spec_path: Path to spec file
        create_backup: Whether to create backup before saving

    Returns:
        True if successful, False otherwise
    """
    if create_backup:
        backup_path = Path(spec_path).with_suffix(".json.backup")
        try:
            with open(spec_path, "r") as f:
                current_data = f.read()
            with open(backup_path, "w") as f:
                f.write(current_data)
        except OSError:
            pass  # Continue even if backup fails

    try:
        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=2)
        return True
    except OSError:
        return False


# Helper functions

def _get_timestamp() -> str:
    """Get current timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clear_journaling_flag(spec_data: Dict[str, Any], task_id: str, timestamp: str) -> None:
    """Clear the needs_journaling flag for a task."""
    hierarchy = spec_data.get("hierarchy", {})
    task = hierarchy.get(task_id)

    if task:
        metadata = task.setdefault("metadata", {})
        if "needs_journaling" in metadata:
            metadata["needs_journaling"] = False
            metadata["journaled_at"] = timestamp


def _recalculate_counts(spec_data: Dict[str, Any]) -> None:
    """Recalculate task counts for all nodes in hierarchy."""
    hierarchy = spec_data.get("hierarchy", {})
    if not hierarchy:
        return

    def calculate_node(node_id: str) -> tuple:
        """Return (total_tasks, completed_tasks) for a node."""
        node = hierarchy.get(node_id, {})
        children = node.get("children", [])
        node_type = node.get("type", "")
        status = node.get("status", "")

        if not children:
            # Leaf node
            if node_type in {"task", "subtask", "verify"}:
                total = 1
                completed = 1 if status == "completed" else 0
            else:
                total = 0
                completed = 0
        else:
            # Parent node: sum children
            total = 0
            completed = 0
            for child_id in children:
                if child_id in hierarchy:
                    child_total, child_completed = calculate_node(child_id)
                    total += child_total
                    completed += child_completed

        node["total_tasks"] = total
        node["completed_tasks"] = completed
        return total, completed

    if "spec-root" in hierarchy:
        calculate_node("spec-root")
