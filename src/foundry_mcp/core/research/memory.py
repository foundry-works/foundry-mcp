"""File-based storage backend for research workflows.

Provides thread-safe persistence for conversation threads, investigation states,
and ideation sessions using file locking.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generic, Optional, TypeVar

from filelock import FileLock

from foundry_mcp.core.research.models import (
    ConsensusState,
    ConversationThread,
    IdeationState,
    ThinkDeepState,
    ThreadStatus,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FileStorageBackend(Generic[T]):
    """Generic file-based storage with locking and TTL support."""

    def __init__(
        self,
        storage_path: Path,
        model_class: type[T],
        ttl_hours: Optional[int] = 24,
    ) -> None:
        """Initialize storage backend.

        Args:
            storage_path: Directory to store files
            model_class: Pydantic model class for serialization
            ttl_hours: Time-to-live in hours (None for no expiry)
        """
        self.storage_path = storage_path
        self.model_class = model_class
        self.ttl_hours = ttl_hours
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, item_id: str) -> Path:
        """Get file path for an item ID."""
        # Sanitize ID to prevent path traversal
        safe_id = "".join(c for c in item_id if c.isalnum() or c in "-_")
        return self.storage_path / f"{safe_id}.json"

    def _get_lock_path(self, item_id: str) -> Path:
        """Get lock file path for an item ID."""
        return self._get_file_path(item_id).with_suffix(".lock")

    def _is_expired(self, file_path: Path) -> bool:
        """Check if a file has expired based on TTL."""
        if self.ttl_hours is None:
            return False

        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            expiry = mtime + timedelta(hours=self.ttl_hours)
            return datetime.now() > expiry
        except OSError:
            return True

    def save(self, item_id: str, item: T) -> None:
        """Save an item to storage with locking.

        Args:
            item_id: Unique identifier for the item
            item: Pydantic model instance to save
        """
        file_path = self._get_file_path(item_id)
        lock_path = self._get_lock_path(item_id)

        with FileLock(lock_path, timeout=10):
            data = item.model_dump(mode="json")
            file_path.write_text(json.dumps(data, indent=2, default=str))
            logger.debug("Saved %s to %s", item_id, file_path)

    def load(self, item_id: str) -> Optional[T]:
        """Load an item from storage with locking.

        Args:
            item_id: Unique identifier for the item

        Returns:
            The loaded item or None if not found/expired
        """
        file_path = self._get_file_path(item_id)
        lock_path = self._get_lock_path(item_id)

        if not file_path.exists():
            return None

        if self._is_expired(file_path):
            logger.debug("Item %s has expired, removing", item_id)
            self.delete(item_id)
            return None

        with FileLock(lock_path, timeout=10):
            try:
                data = json.loads(file_path.read_text())
                return self.model_class.model_validate(data)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to load %s: %s", item_id, exc)
                return None

    def delete(self, item_id: str) -> bool:
        """Delete an item from storage.

        Args:
            item_id: Unique identifier for the item

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_file_path(item_id)
        lock_path = self._get_lock_path(item_id)

        if not file_path.exists():
            return False

        with FileLock(lock_path, timeout=10):
            try:
                file_path.unlink()
                logger.debug("Deleted %s", item_id)
                # Clean up lock file
                if lock_path.exists():
                    lock_path.unlink()
                return True
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", item_id, exc)
                return False

    def list_ids(self) -> list[str]:
        """List all item IDs in storage.

        Returns:
            List of item IDs (without .json extension)
        """
        if not self.storage_path.exists():
            return []

        ids = []
        for file_path in self.storage_path.glob("*.json"):
            item_id = file_path.stem
            # Skip expired items
            if not self._is_expired(file_path):
                ids.append(item_id)
        return sorted(ids)

    def cleanup_expired(self) -> int:
        """Remove all expired items from storage.

        Returns:
            Number of items removed
        """
        if self.ttl_hours is None:
            return 0

        removed = 0
        for file_path in self.storage_path.glob("*.json"):
            if self._is_expired(file_path):
                item_id = file_path.stem
                if self.delete(item_id):
                    removed += 1
        return removed


class ResearchMemory:
    """Unified memory interface for all research workflow states.

    Provides CRUD operations for conversation threads, investigation states,
    ideation sessions, and consensus states.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        ttl_hours: int = 24,
    ) -> None:
        """Initialize research memory.

        Args:
            base_path: Base directory for all storage (default: ~/.foundry-mcp/research)
            ttl_hours: Default TTL for all storages
        """
        if base_path is None:
            base_path = Path.home() / ".foundry-mcp" / "research"

        self.base_path = base_path
        self.ttl_hours = ttl_hours

        # Initialize storage backends for each type
        self._threads = FileStorageBackend(
            storage_path=base_path / "threads",
            model_class=ConversationThread,
            ttl_hours=ttl_hours,
        )
        self._investigations = FileStorageBackend(
            storage_path=base_path / "investigations",
            model_class=ThinkDeepState,
            ttl_hours=ttl_hours,
        )
        self._ideations = FileStorageBackend(
            storage_path=base_path / "ideations",
            model_class=IdeationState,
            ttl_hours=ttl_hours,
        )
        self._consensus = FileStorageBackend(
            storage_path=base_path / "consensus",
            model_class=ConsensusState,
            ttl_hours=ttl_hours,
        )

    # =========================================================================
    # Thread operations (CHAT workflow)
    # =========================================================================

    def save_thread(self, thread: ConversationThread) -> None:
        """Save a conversation thread."""
        self._threads.save(thread.id, thread)

    def load_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """Load a conversation thread by ID."""
        return self._threads.load(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread."""
        return self._threads.delete(thread_id)

    def list_threads(
        self,
        status: Optional[ThreadStatus] = None,
        limit: Optional[int] = None,
    ) -> list[ConversationThread]:
        """List conversation threads, optionally filtered by status.

        Args:
            status: Filter by thread status
            limit: Maximum number of threads to return

        Returns:
            List of conversation threads
        """
        threads = []
        for thread_id in self._threads.list_ids():
            thread = self._threads.load(thread_id)
            if thread is not None:
                if status is None or thread.status == status:
                    threads.append(thread)

        # Sort by updated_at descending
        threads.sort(key=lambda t: t.updated_at, reverse=True)

        if limit is not None:
            threads = threads[:limit]

        return threads

    # =========================================================================
    # Investigation operations (THINKDEEP workflow)
    # =========================================================================

    def save_investigation(self, investigation: ThinkDeepState) -> None:
        """Save an investigation state."""
        self._investigations.save(investigation.id, investigation)

    def load_investigation(self, investigation_id: str) -> Optional[ThinkDeepState]:
        """Load an investigation state by ID."""
        return self._investigations.load(investigation_id)

    def delete_investigation(self, investigation_id: str) -> bool:
        """Delete an investigation state."""
        return self._investigations.delete(investigation_id)

    def list_investigations(
        self,
        limit: Optional[int] = None,
    ) -> list[ThinkDeepState]:
        """List investigation states.

        Args:
            limit: Maximum number of investigations to return

        Returns:
            List of investigation states
        """
        investigations = []
        for inv_id in self._investigations.list_ids():
            inv = self._investigations.load(inv_id)
            if inv is not None:
                investigations.append(inv)

        # Sort by updated_at descending
        investigations.sort(key=lambda i: i.updated_at, reverse=True)

        if limit is not None:
            investigations = investigations[:limit]

        return investigations

    # =========================================================================
    # Ideation operations (IDEATE workflow)
    # =========================================================================

    def save_ideation(self, ideation: IdeationState) -> None:
        """Save an ideation state."""
        self._ideations.save(ideation.id, ideation)

    def load_ideation(self, ideation_id: str) -> Optional[IdeationState]:
        """Load an ideation state by ID."""
        return self._ideations.load(ideation_id)

    def delete_ideation(self, ideation_id: str) -> bool:
        """Delete an ideation state."""
        return self._ideations.delete(ideation_id)

    def list_ideations(
        self,
        limit: Optional[int] = None,
    ) -> list[IdeationState]:
        """List ideation states.

        Args:
            limit: Maximum number of ideations to return

        Returns:
            List of ideation states
        """
        ideations = []
        for ide_id in self._ideations.list_ids():
            ide = self._ideations.load(ide_id)
            if ide is not None:
                ideations.append(ide)

        # Sort by updated_at descending
        ideations.sort(key=lambda i: i.updated_at, reverse=True)

        if limit is not None:
            ideations = ideations[:limit]

        return ideations

    # =========================================================================
    # Consensus operations (CONSENSUS workflow)
    # =========================================================================

    def save_consensus(self, consensus: ConsensusState) -> None:
        """Save a consensus state."""
        self._consensus.save(consensus.id, consensus)

    def load_consensus(self, consensus_id: str) -> Optional[ConsensusState]:
        """Load a consensus state by ID."""
        return self._consensus.load(consensus_id)

    def delete_consensus(self, consensus_id: str) -> bool:
        """Delete a consensus state."""
        return self._consensus.delete(consensus_id)

    def list_consensus(
        self,
        limit: Optional[int] = None,
    ) -> list[ConsensusState]:
        """List consensus states.

        Args:
            limit: Maximum number of consensus states to return

        Returns:
            List of consensus states
        """
        states = []
        for cons_id in self._consensus.list_ids():
            cons = self._consensus.load(cons_id)
            if cons is not None:
                states.append(cons)

        # Sort by created_at descending
        states.sort(key=lambda c: c.created_at, reverse=True)

        if limit is not None:
            states = states[:limit]

        return states

    # =========================================================================
    # Maintenance operations
    # =========================================================================

    def cleanup_all_expired(self) -> dict[str, int]:
        """Remove expired items from all storages.

        Returns:
            Dict with counts of removed items per storage type
        """
        return {
            "threads": self._threads.cleanup_expired(),
            "investigations": self._investigations.cleanup_expired(),
            "ideations": self._ideations.cleanup_expired(),
            "consensus": self._consensus.cleanup_expired(),
        }

    def get_storage_stats(self) -> dict[str, int]:
        """Get count of items in each storage.

        Returns:
            Dict with counts per storage type
        """
        return {
            "threads": len(self._threads.list_ids()),
            "investigations": len(self._investigations.list_ids()),
            "ideations": len(self._ideations.list_ids()),
            "consensus": len(self._consensus.list_ids()),
        }
