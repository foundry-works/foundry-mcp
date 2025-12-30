"""Unit tests for research workflow storage backend and memory.

Tests FileStorageBackend for generic storage operations and ResearchMemory
for unified CRUD operations on threads, investigations, ideations, and consensus.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from foundry_mcp.core.research.memory import FileStorageBackend, ResearchMemory
from foundry_mcp.core.research.models import (
    ConsensusConfig,
    ConsensusState,
    ConsensusStrategy,
    ConversationThread,
    IdeationState,
    ThinkDeepState,
    ThreadStatus,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleModel(BaseModel):
    """Simple model for testing FileStorageBackend."""

    id: str
    name: str
    value: int = 0


@pytest.fixture
def temp_storage_path(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage_path = tmp_path / "test_storage"
    storage_path.mkdir()
    return storage_path


@pytest.fixture
def storage_backend(temp_storage_path: Path) -> FileStorageBackend[SimpleModel]:
    """Create a FileStorageBackend for testing."""
    return FileStorageBackend(
        storage_path=temp_storage_path,
        model_class=SimpleModel,
        ttl_hours=24,
    )


@pytest.fixture
def research_memory(tmp_path: Path) -> ResearchMemory:
    """Create a ResearchMemory instance for testing."""
    base_path = tmp_path / "research_memory"
    return ResearchMemory(base_path=base_path, ttl_hours=24)


# =============================================================================
# FileStorageBackend Tests
# =============================================================================


class TestFileStorageBackendInit:
    """Tests for FileStorageBackend initialization."""

    def test_creates_directory(self, tmp_path: Path):
        """Should create storage directory if it doesn't exist."""
        storage_path = tmp_path / "new_storage"
        assert not storage_path.exists()

        FileStorageBackend(
            storage_path=storage_path,
            model_class=SimpleModel,
            ttl_hours=24,
        )

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_uses_existing_directory(self, temp_storage_path: Path):
        """Should work with existing directory."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=24,
        )

        assert backend.storage_path == temp_storage_path

    def test_ttl_none_disables_expiry(self, temp_storage_path: Path):
        """Should accept None TTL to disable expiry."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=None,
        )

        assert backend.ttl_hours is None


class TestFileStorageBackendPathSanitization:
    """Tests for path sanitization in FileStorageBackend."""

    def test_safe_path_generation(self, storage_backend: FileStorageBackend):
        """Should generate safe file paths."""
        # Normal ID
        path = storage_backend._get_file_path("test-item-123")
        assert path.name == "test-item-123.json"

        # ID with underscores
        path = storage_backend._get_file_path("test_item_456")
        assert path.name == "test_item_456.json"

    def test_sanitizes_path_traversal(self, storage_backend: FileStorageBackend):
        """Should sanitize path traversal attempts."""
        # Path traversal attempt
        path = storage_backend._get_file_path("../../../etc/passwd")
        assert ".." not in str(path)
        assert "etcpasswd.json" in path.name

    def test_sanitizes_special_characters(self, storage_backend: FileStorageBackend):
        """Should remove special characters from IDs."""
        path = storage_backend._get_file_path("test<>:\"\\|?*item")
        assert "<" not in path.name
        assert ">" not in path.name
        assert "testitem.json" == path.name


class TestFileStorageBackendCRUD:
    """Tests for CRUD operations on FileStorageBackend."""

    def test_save_creates_file(self, storage_backend: FileStorageBackend):
        """Should create JSON file on save."""
        item = SimpleModel(id="item-1", name="Test", value=42)
        storage_backend.save("item-1", item)

        file_path = storage_backend._get_file_path("item-1")
        assert file_path.exists()

    def test_load_returns_item(self, storage_backend: FileStorageBackend):
        """Should load saved item correctly."""
        item = SimpleModel(id="item-1", name="Test", value=42)
        storage_backend.save("item-1", item)

        loaded = storage_backend.load("item-1")

        assert loaded is not None
        assert loaded.id == "item-1"
        assert loaded.name == "Test"
        assert loaded.value == 42

    def test_load_nonexistent_returns_none(self, storage_backend: FileStorageBackend):
        """Should return None for nonexistent item."""
        loaded = storage_backend.load("nonexistent")
        assert loaded is None

    def test_load_invalid_json_returns_none(
        self, storage_backend: FileStorageBackend, temp_storage_path: Path
    ):
        """Should return None for invalid JSON."""
        # Create invalid JSON file
        file_path = temp_storage_path / "invalid.json"
        file_path.write_text("not valid json {{{")

        loaded = storage_backend.load("invalid")
        assert loaded is None

    def test_delete_removes_file(self, storage_backend: FileStorageBackend):
        """Should delete file and return True."""
        item = SimpleModel(id="item-1", name="Test")
        storage_backend.save("item-1", item)

        result = storage_backend.delete("item-1")

        assert result is True
        assert not storage_backend._get_file_path("item-1").exists()

    def test_delete_nonexistent_returns_false(
        self, storage_backend: FileStorageBackend
    ):
        """Should return False when deleting nonexistent item."""
        result = storage_backend.delete("nonexistent")
        assert result is False

    def test_list_ids_returns_all(self, storage_backend: FileStorageBackend):
        """Should list all item IDs."""
        storage_backend.save("item-a", SimpleModel(id="item-a", name="A"))
        storage_backend.save("item-b", SimpleModel(id="item-b", name="B"))
        storage_backend.save("item-c", SimpleModel(id="item-c", name="C"))

        ids = storage_backend.list_ids()

        assert len(ids) == 3
        assert "item-a" in ids
        assert "item-b" in ids
        assert "item-c" in ids

    def test_list_ids_sorted(self, storage_backend: FileStorageBackend):
        """Should return sorted IDs."""
        storage_backend.save("z-item", SimpleModel(id="z", name="Z"))
        storage_backend.save("a-item", SimpleModel(id="a", name="A"))
        storage_backend.save("m-item", SimpleModel(id="m", name="M"))

        ids = storage_backend.list_ids()

        assert ids == ["a-item", "m-item", "z-item"]

    def test_list_ids_empty_storage(self, storage_backend: FileStorageBackend):
        """Should return empty list for empty storage."""
        ids = storage_backend.list_ids()
        assert ids == []

    def test_update_overwrites(self, storage_backend: FileStorageBackend):
        """Should overwrite existing item."""
        item1 = SimpleModel(id="item-1", name="Original", value=1)
        storage_backend.save("item-1", item1)

        item2 = SimpleModel(id="item-1", name="Updated", value=2)
        storage_backend.save("item-1", item2)

        loaded = storage_backend.load("item-1")
        assert loaded.name == "Updated"
        assert loaded.value == 2


class TestFileStorageBackendTTL:
    """Tests for TTL functionality in FileStorageBackend."""

    def test_is_expired_false_within_ttl(
        self, storage_backend: FileStorageBackend, temp_storage_path: Path
    ):
        """Should return False for non-expired items."""
        file_path = temp_storage_path / "fresh.json"
        file_path.write_text('{"id": "fresh", "name": "Test"}')

        assert storage_backend._is_expired(file_path) is False

    def test_is_expired_true_past_ttl(self, temp_storage_path: Path):
        """Should return True for expired items."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=1,  # 1 hour TTL
        )

        file_path = temp_storage_path / "old.json"
        file_path.write_text('{"id": "old", "name": "Test"}')

        # Mock file mtime to be 2 hours ago
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(file_path, (old_time, old_time))

        assert backend._is_expired(file_path) is True

    def test_is_expired_none_ttl_never_expires(self, temp_storage_path: Path):
        """Should never expire when TTL is None."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=None,
        )

        file_path = temp_storage_path / "permanent.json"
        file_path.write_text('{"id": "permanent", "name": "Test"}')

        # Set file to be very old
        old_time = (datetime.now() - timedelta(days=365)).timestamp()
        os.utime(file_path, (old_time, old_time))

        assert backend._is_expired(file_path) is False

    def test_load_deletes_expired(self, temp_storage_path: Path):
        """Should delete expired item on load and return None."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=1,
        )

        # Save item
        item = SimpleModel(id="expiring", name="Test")
        backend.save("expiring", item)

        # Make it expired
        file_path = backend._get_file_path("expiring")
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(file_path, (old_time, old_time))

        # Load should return None and delete
        loaded = backend.load("expiring")

        assert loaded is None
        assert not file_path.exists()

    def test_cleanup_expired_removes_old_items(self, temp_storage_path: Path):
        """Should remove all expired items."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=1,
        )

        # Save fresh and expired items
        backend.save("fresh", SimpleModel(id="fresh", name="Fresh"))
        backend.save("old-1", SimpleModel(id="old-1", name="Old 1"))
        backend.save("old-2", SimpleModel(id="old-2", name="Old 2"))

        # Make some items expired
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(backend._get_file_path("old-1"), (old_time, old_time))
        os.utime(backend._get_file_path("old-2"), (old_time, old_time))

        removed = backend.cleanup_expired()

        assert removed == 2
        assert backend.load("fresh") is not None
        assert backend.load("old-1") is None
        assert backend.load("old-2") is None

    def test_cleanup_expired_none_ttl_removes_nothing(self, temp_storage_path: Path):
        """Should remove nothing when TTL is None."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=None,
        )

        backend.save("item", SimpleModel(id="item", name="Test"))

        removed = backend.cleanup_expired()

        assert removed == 0

    def test_list_ids_excludes_expired(self, temp_storage_path: Path):
        """Should exclude expired items from list."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=1,
        )

        backend.save("fresh", SimpleModel(id="fresh", name="Fresh"))
        backend.save("expired", SimpleModel(id="expired", name="Expired"))

        # Make one item expired
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(backend._get_file_path("expired"), (old_time, old_time))

        ids = backend.list_ids()

        assert "fresh" in ids
        assert "expired" not in ids


class TestFileStorageBackendConcurrency:
    """Tests for concurrent access to FileStorageBackend."""

    def test_concurrent_saves(self, temp_storage_path: Path):
        """Should handle concurrent saves with locking."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=24,
        )

        def save_item(value: int) -> bool:
            item = SimpleModel(id="shared", name=f"Value-{value}", value=value)
            backend.save("shared", item)
            return True

        # Run concurrent saves
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(save_item, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]

        # All saves should complete
        assert all(results)

        # Final item should be valid
        loaded = backend.load("shared")
        assert loaded is not None
        assert loaded.name.startswith("Value-")

    def test_concurrent_reads(self, storage_backend: FileStorageBackend):
        """Should handle concurrent reads."""
        item = SimpleModel(id="read-test", name="Test", value=100)
        storage_backend.save("read-test", item)

        def read_item() -> Optional[SimpleModel]:
            return storage_backend.load("read-test")

        # Run concurrent reads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(read_item) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]

        # All reads should succeed
        assert all(r is not None for r in results)
        assert all(r.value == 100 for r in results)

    def test_concurrent_read_write(self, temp_storage_path: Path):
        """Should handle concurrent reads and writes."""
        backend = FileStorageBackend(
            storage_path=temp_storage_path,
            model_class=SimpleModel,
            ttl_hours=24,
        )

        # Initialize item
        backend.save("mixed", SimpleModel(id="mixed", name="Initial", value=0))

        def write_item(value: int) -> bool:
            item = SimpleModel(id="mixed", name=f"Write-{value}", value=value)
            backend.save("mixed", item)
            return True

        def read_item() -> bool:
            result = backend.load("mixed")
            return result is not None

        # Run mixed concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            write_futures = [executor.submit(write_item, i) for i in range(10)]
            read_futures = [executor.submit(read_item) for _ in range(20)]

            all_futures = write_futures + read_futures
            results = [f.result() for f in as_completed(all_futures)]

        # All operations should succeed
        assert all(results)


# =============================================================================
# ResearchMemory Tests
# =============================================================================


class TestResearchMemoryInit:
    """Tests for ResearchMemory initialization."""

    def test_creates_storage_directories(self, tmp_path: Path):
        """Should create all storage subdirectories."""
        base_path = tmp_path / "research"
        memory = ResearchMemory(base_path=base_path)

        assert (base_path / "threads").exists()
        assert (base_path / "investigations").exists()
        assert (base_path / "ideations").exists()
        assert (base_path / "consensus").exists()

    def test_default_path(self, tmp_path: Path):
        """Should use default path when none provided."""
        mock_home = tmp_path / "mock_home"
        mock_home.mkdir()

        with patch.object(Path, "home", return_value=mock_home):
            memory = ResearchMemory()

            assert memory.base_path == mock_home / ".foundry-mcp" / "research"

    def test_custom_ttl(self, tmp_path: Path):
        """Should accept custom TTL."""
        memory = ResearchMemory(base_path=tmp_path, ttl_hours=48)
        assert memory.ttl_hours == 48


class TestResearchMemoryThreads:
    """Tests for thread operations in ResearchMemory."""

    def test_save_and_load_thread(self, research_memory: ResearchMemory):
        """Should save and load threads correctly."""
        thread = ConversationThread(title="Test Thread")
        thread.add_message(role="user", content="Hello")

        research_memory.save_thread(thread)
        loaded = research_memory.load_thread(thread.id)

        assert loaded is not None
        assert loaded.title == "Test Thread"
        assert len(loaded.messages) == 1

    def test_delete_thread(self, research_memory: ResearchMemory):
        """Should delete threads correctly."""
        thread = ConversationThread()
        research_memory.save_thread(thread)

        result = research_memory.delete_thread(thread.id)

        assert result is True
        assert research_memory.load_thread(thread.id) is None

    def test_list_threads(self, research_memory: ResearchMemory):
        """Should list all threads."""
        thread1 = ConversationThread(title="Thread 1")
        thread2 = ConversationThread(title="Thread 2")
        thread3 = ConversationThread(title="Thread 3")

        research_memory.save_thread(thread1)
        research_memory.save_thread(thread2)
        research_memory.save_thread(thread3)

        threads = research_memory.list_threads()

        assert len(threads) == 3

    def test_list_threads_by_status(self, research_memory: ResearchMemory):
        """Should filter threads by status."""
        active = ConversationThread(status=ThreadStatus.ACTIVE)
        completed = ConversationThread(status=ThreadStatus.COMPLETED)
        archived = ConversationThread(status=ThreadStatus.ARCHIVED)

        research_memory.save_thread(active)
        research_memory.save_thread(completed)
        research_memory.save_thread(archived)

        active_threads = research_memory.list_threads(status=ThreadStatus.ACTIVE)
        completed_threads = research_memory.list_threads(status=ThreadStatus.COMPLETED)

        assert len(active_threads) == 1
        assert len(completed_threads) == 1
        assert active_threads[0].id == active.id

    def test_list_threads_with_limit(self, research_memory: ResearchMemory):
        """Should respect limit parameter."""
        for i in range(10):
            thread = ConversationThread(title=f"Thread {i}")
            research_memory.save_thread(thread)

        threads = research_memory.list_threads(limit=3)

        assert len(threads) == 3

    def test_list_threads_sorted_by_updated_at(self, research_memory: ResearchMemory):
        """Should return threads sorted by updated_at descending."""
        thread1 = ConversationThread(title="Thread 1")
        thread2 = ConversationThread(title="Thread 2")

        research_memory.save_thread(thread1)
        time.sleep(0.01)  # Ensure different timestamps
        research_memory.save_thread(thread2)

        threads = research_memory.list_threads()

        # Most recently updated first
        assert threads[0].title == "Thread 2"
        assert threads[1].title == "Thread 1"


class TestResearchMemoryInvestigations:
    """Tests for investigation operations in ResearchMemory."""

    def test_save_and_load_investigation(self, research_memory: ResearchMemory):
        """Should save and load investigations correctly."""
        investigation = ThinkDeepState(topic="Test Topic", max_depth=3)
        investigation.add_hypothesis("Test hypothesis")

        research_memory.save_investigation(investigation)
        loaded = research_memory.load_investigation(investigation.id)

        assert loaded is not None
        assert loaded.topic == "Test Topic"
        assert loaded.max_depth == 3
        assert len(loaded.hypotheses) == 1

    def test_delete_investigation(self, research_memory: ResearchMemory):
        """Should delete investigations correctly."""
        investigation = ThinkDeepState(topic="Test")
        research_memory.save_investigation(investigation)

        result = research_memory.delete_investigation(investigation.id)

        assert result is True
        assert research_memory.load_investigation(investigation.id) is None

    def test_list_investigations(self, research_memory: ResearchMemory):
        """Should list investigations with limit."""
        for i in range(5):
            inv = ThinkDeepState(topic=f"Topic {i}")
            research_memory.save_investigation(inv)

        investigations = research_memory.list_investigations(limit=3)

        assert len(investigations) == 3


class TestResearchMemoryIdeations:
    """Tests for ideation operations in ResearchMemory."""

    def test_save_and_load_ideation(self, research_memory: ResearchMemory):
        """Should save and load ideations correctly."""
        ideation = IdeationState(topic="New Feature")
        ideation.add_idea("Great idea", perspective="technical")

        research_memory.save_ideation(ideation)
        loaded = research_memory.load_ideation(ideation.id)

        assert loaded is not None
        assert loaded.topic == "New Feature"
        assert len(loaded.ideas) == 1

    def test_delete_ideation(self, research_memory: ResearchMemory):
        """Should delete ideations correctly."""
        ideation = IdeationState(topic="Test")
        research_memory.save_ideation(ideation)

        result = research_memory.delete_ideation(ideation.id)

        assert result is True
        assert research_memory.load_ideation(ideation.id) is None

    def test_list_ideations(self, research_memory: ResearchMemory):
        """Should list ideations with limit."""
        for i in range(5):
            ide = IdeationState(topic=f"Topic {i}")
            research_memory.save_ideation(ide)

        ideations = research_memory.list_ideations(limit=2)

        assert len(ideations) == 2


class TestResearchMemoryConsensus:
    """Tests for consensus operations in ResearchMemory."""

    def test_save_and_load_consensus(self, research_memory: ResearchMemory):
        """Should save and load consensus correctly."""
        config = ConsensusConfig(
            providers=["openai", "anthropic"],
            strategy=ConsensusStrategy.SYNTHESIZE,
        )
        consensus = ConsensusState(prompt="Test prompt", config=config)

        research_memory.save_consensus(consensus)
        loaded = research_memory.load_consensus(consensus.id)

        assert loaded is not None
        assert loaded.prompt == "Test prompt"
        assert len(loaded.config.providers) == 2

    def test_delete_consensus(self, research_memory: ResearchMemory):
        """Should delete consensus correctly."""
        config = ConsensusConfig(providers=["openai"])
        consensus = ConsensusState(prompt="Test", config=config)
        research_memory.save_consensus(consensus)

        result = research_memory.delete_consensus(consensus.id)

        assert result is True
        assert research_memory.load_consensus(consensus.id) is None

    def test_list_consensus(self, research_memory: ResearchMemory):
        """Should list consensus states with limit."""
        for i in range(5):
            config = ConsensusConfig(providers=["openai"])
            cons = ConsensusState(prompt=f"Prompt {i}", config=config)
            research_memory.save_consensus(cons)

        states = research_memory.list_consensus(limit=3)

        assert len(states) == 3


class TestResearchMemoryMaintenance:
    """Tests for maintenance operations in ResearchMemory."""

    def test_cleanup_all_expired(self, tmp_path: Path):
        """Should cleanup expired items from all storages."""
        memory = ResearchMemory(base_path=tmp_path, ttl_hours=1)

        # Save items in each storage
        thread = ConversationThread()
        investigation = ThinkDeepState(topic="Test")
        ideation = IdeationState(topic="Test")
        config = ConsensusConfig(providers=["openai"])
        consensus = ConsensusState(prompt="Test", config=config)

        memory.save_thread(thread)
        memory.save_investigation(investigation)
        memory.save_ideation(ideation)
        memory.save_consensus(consensus)

        # Make all items expired
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        for storage_dir in ["threads", "investigations", "ideations", "consensus"]:
            for file_path in (tmp_path / storage_dir).glob("*.json"):
                os.utime(file_path, (old_time, old_time))

        result = memory.cleanup_all_expired()

        assert result["threads"] == 1
        assert result["investigations"] == 1
        assert result["ideations"] == 1
        assert result["consensus"] == 1

    def test_get_storage_stats(self, research_memory: ResearchMemory):
        """Should return counts per storage type."""
        # Add items
        research_memory.save_thread(ConversationThread())
        research_memory.save_thread(ConversationThread())
        research_memory.save_investigation(ThinkDeepState(topic="Test"))
        config = ConsensusConfig(providers=["openai"])
        research_memory.save_consensus(
            ConsensusState(prompt="Test", config=config)
        )

        stats = research_memory.get_storage_stats()

        assert stats["threads"] == 2
        assert stats["investigations"] == 1
        assert stats["ideations"] == 0
        assert stats["consensus"] == 1

    def test_get_storage_stats_empty(self, research_memory: ResearchMemory):
        """Should return zeros for empty storage."""
        stats = research_memory.get_storage_stats()

        assert stats["threads"] == 0
        assert stats["investigations"] == 0
        assert stats["ideations"] == 0
        assert stats["consensus"] == 0


class TestResearchMemoryConcurrency:
    """Tests for concurrent access to ResearchMemory."""

    def test_concurrent_thread_operations(self, tmp_path: Path):
        """Should handle concurrent thread operations."""
        memory = ResearchMemory(base_path=tmp_path, ttl_hours=24)

        def create_and_update_thread(index: int) -> bool:
            thread = ConversationThread(title=f"Thread {index}")
            thread.add_message(role="user", content=f"Message {index}")
            memory.save_thread(thread)
            loaded = memory.load_thread(thread.id)
            return loaded is not None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(create_and_update_thread, i) for i in range(20)
            ]
            results = [f.result() for f in as_completed(futures)]

        assert all(results)
        assert len(memory.list_threads()) == 20

    def test_concurrent_mixed_storage_operations(self, tmp_path: Path):
        """Should handle concurrent operations across storage types."""
        memory = ResearchMemory(base_path=tmp_path, ttl_hours=24)

        def thread_op(index: int) -> bool:
            thread = ConversationThread(title=f"T{index}")
            memory.save_thread(thread)
            return memory.load_thread(thread.id) is not None

        def investigation_op(index: int) -> bool:
            inv = ThinkDeepState(topic=f"I{index}")
            memory.save_investigation(inv)
            return memory.load_investigation(inv.id) is not None

        def ideation_op(index: int) -> bool:
            ide = IdeationState(topic=f"ID{index}")
            memory.save_ideation(ide)
            return memory.load_ideation(ide.id) is not None

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for i in range(10):
                futures.append(executor.submit(thread_op, i))
                futures.append(executor.submit(investigation_op, i))
                futures.append(executor.submit(ideation_op, i))

            results = [f.result() for f in as_completed(futures)]

        assert all(results)

        stats = memory.get_storage_stats()
        assert stats["threads"] == 10
        assert stats["investigations"] == 10
        assert stats["ideations"] == 10
