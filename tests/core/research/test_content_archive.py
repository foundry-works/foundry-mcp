"""Tests for content archive storage.

Tests cover:
1. ArchivedContent serialization - to_dict/from_dict roundtrip
2. Content hash computation - SHA256 hashing
3. Basic archive operations - archive/retrieve cycle
4. TTL cleanup - expiry detection and cleanup
5. Private permissions - directory (0o700) and file (0o600)
6. Corrupted JSON handling - skip-on-corruption policy
7. Read-only filesystem - graceful handling
8. Atomic writes - temp file + rename pattern
9. Guardrails - enabled/disabled state, warnings
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from foundry_mcp.core.research.content_archive import (
    ARCHIVE_READ_CORRUPT,
    ARCHIVE_WRITE_FAILED,
    DEFAULT_ARCHIVE_TTL_HOURS,
    ArchivedContent,
    ContentArchive,
    compute_content_hash,
)

# =============================================================================
# Test: ArchivedContent Serialization
# =============================================================================


class TestArchivedContentSerialization:
    """Tests for ArchivedContent to_dict/from_dict roundtrip."""

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all expected fields."""
        now = datetime.now(timezone.utc)
        record = ArchivedContent(
            content_hash="abc123",
            content="Test content",
            item_id="item-1",
            item_type="source",
            archived_at=now,
            archive_reason="budget_exceeded",
            original_tokens=100,
            metadata={"key": "value"},
        )
        data = record.to_dict()

        assert data["content_hash"] == "abc123"
        assert data["content"] == "Test content"
        assert data["item_id"] == "item-1"
        assert data["item_type"] == "source"
        assert data["archived_at"] == now.isoformat()
        assert data["archive_reason"] == "budget_exceeded"
        assert data["original_tokens"] == 100
        assert data["metadata"] == {"key": "value"}

    def test_from_dict_roundtrip(self):
        """Test from_dict correctly deserializes to_dict output."""
        original = ArchivedContent(
            content_hash="abc123",
            content="Test content",
            item_id="item-1",
            item_type="finding",
            archive_reason="compressed",
            original_tokens=50,
            metadata={"source": "test"},
        )
        data = original.to_dict()
        restored = ArchivedContent.from_dict(data)

        assert restored.content_hash == original.content_hash
        assert restored.content == original.content
        assert restored.item_id == original.item_id
        assert restored.item_type == original.item_type
        assert restored.archive_reason == original.archive_reason
        assert restored.original_tokens == original.original_tokens
        assert restored.metadata == original.metadata

    def test_from_dict_handles_iso_timestamp(self):
        """Test from_dict parses ISO format timestamps."""
        data = {
            "content_hash": "abc",
            "content": "test",
            "item_id": "id",
            "archived_at": "2024-01-15T10:30:00+00:00",
        }
        record = ArchivedContent.from_dict(data)
        assert record.archived_at.year == 2024
        assert record.archived_at.month == 1
        assert record.archived_at.day == 15

    def test_from_dict_handles_z_suffix(self):
        """Test from_dict handles Z suffix in timestamps."""
        data = {
            "content_hash": "abc",
            "content": "test",
            "item_id": "id",
            "archived_at": "2024-01-15T10:30:00Z",
        }
        record = ArchivedContent.from_dict(data)
        assert record.archived_at.tzinfo is not None

    def test_from_dict_handles_missing_optional_fields(self):
        """Test from_dict uses defaults for missing optional fields."""
        data = {
            "content_hash": "abc",
            "content": "test",
            "item_id": "id",
        }
        record = ArchivedContent.from_dict(data)
        assert record.item_type == "source"
        assert record.archive_reason == ""
        assert record.original_tokens is None
        assert record.metadata == {}


# =============================================================================
# Test: Content Hash Computation
# =============================================================================


class TestComputeContentHash:
    """Tests for SHA256 hash computation."""

    def test_returns_hex_string(self):
        """Test hash is returned as hex string."""
        result = compute_content_hash("test")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_returns_64_chars(self):
        """Test SHA256 hash is 64 characters."""
        result = compute_content_hash("test content")
        assert len(result) == 64

    def test_same_content_same_hash(self):
        """Test identical content produces identical hash."""
        hash1 = compute_content_hash("same content")
        hash2 = compute_content_hash("same content")
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        hash1 = compute_content_hash("content one")
        hash2 = compute_content_hash("content two")
        assert hash1 != hash2

    def test_handles_unicode(self):
        """Test hash handles unicode content."""
        result = compute_content_hash("Unicode: \u00e9\u00e0\u00fc\u4e2d\u6587")
        assert len(result) == 64

    def test_handles_empty_string(self):
        """Test hash handles empty string."""
        result = compute_content_hash("")
        assert len(result) == 64


# =============================================================================
# Test: Basic Archive Operations
# =============================================================================


class TestContentArchiveBasicOperations:
    """Tests for archive/retrieve cycle."""

    @pytest.fixture
    def archive(self, tmp_path):
        """Create archive instance for testing."""
        return ContentArchive(storage_path=tmp_path, enabled=True)

    def test_archive_returns_record(self, archive):
        """Test archive returns ArchivedContent record."""
        result = archive.archive(
            content="Test content",
            item_id="item-1",
            reason="test",
        )
        assert isinstance(result, ArchivedContent)
        assert result.content == "Test content"
        assert result.item_id == "item-1"
        assert result.archive_reason == "test"

    def test_archive_creates_file(self, archive, tmp_path):
        """Test archive creates JSON file."""
        result = archive.archive(
            content="Test content",
            item_id="item-1",
        )
        file_path = tmp_path / f"{result.content_hash}.json"
        assert file_path.exists()

    def test_retrieve_returns_archived_content(self, archive):
        """Test retrieve returns previously archived content."""
        archived = archive.archive(
            content="Test content",
            item_id="item-1",
        )
        retrieved = archive.retrieve(archived.content_hash)
        assert retrieved is not None
        assert retrieved.content == "Test content"
        assert retrieved.item_id == "item-1"

    def test_retrieve_returns_none_for_unknown_hash(self, archive):
        """Test retrieve returns None for unknown hash."""
        result = archive.retrieve("a" * 64)
        assert result is None

    def test_deduplication_preserves_original_timestamp(self, archive):
        """Test archiving same content preserves original timestamp."""
        first = archive.archive(
            content="Same content",
            item_id="item-1",
        )
        # Archive same content again
        second = archive.archive(
            content="Same content",
            item_id="item-2",
            reason="updated",
        )
        # Should have same hash
        assert first.content_hash == second.content_hash
        # Timestamp should be preserved from first archive
        assert second.archived_at == first.archived_at

    def test_retrieve_by_item_id(self, archive):
        """Test retrieve_by_item_id finds matching records."""
        archive.archive(content="Content 1", item_id="target-item")
        archive.archive(content="Content 2", item_id="other-item")
        archive.archive(content="Content 3", item_id="target-item")

        results = archive.retrieve_by_item_id("target-item")
        assert len(results) == 2
        assert all(r.item_id == "target-item" for r in results)

    def test_delete_removes_file(self, archive, tmp_path):
        """Test delete removes archive file."""
        archived = archive.archive(content="Test", item_id="item-1")
        file_path = tmp_path / f"{archived.content_hash}.json"
        assert file_path.exists()

        result = archive.delete(archived.content_hash)
        assert result is True
        assert not file_path.exists()

    def test_delete_returns_false_for_unknown(self, archive):
        """Test delete returns False for unknown hash."""
        result = archive.delete("a" * 64)
        assert result is False

    def test_list_hashes(self, archive):
        """Test list_hashes returns all content hashes."""
        archive.archive(content="Content 1", item_id="item-1")
        archive.archive(content="Content 2", item_id="item-2")

        hashes = archive.list_hashes()
        assert len(hashes) == 2


# =============================================================================
# Test: TTL Cleanup
# =============================================================================


class TestContentArchiveTTLCleanup:
    """Tests for TTL expiry and cleanup."""

    def test_expired_content_not_retrieved(self, tmp_path):
        """Test expired content returns None on retrieve."""
        # Create archive with very short TTL
        archive = ContentArchive(storage_path=tmp_path, ttl_hours=0, enabled=True)
        archived = archive.archive(content="Test", item_id="item-1")

        # Mock time to make content expired
        with patch.object(archive, "_is_expired", return_value=True):
            result = archive.retrieve(archived.content_hash)
            assert result is None

    def test_cleanup_expired_removes_old_files(self, tmp_path):
        """Test cleanup_expired removes expired files."""
        archive = ContentArchive(storage_path=tmp_path, ttl_hours=1, enabled=True)
        archived = archive.archive(content="Test", item_id="item-1")

        # Make the file appear old by modifying mtime
        file_path = tmp_path / f"{archived.content_hash}.json"
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(file_path, (old_time, old_time))

        removed = archive.cleanup_expired()
        assert removed == 1
        assert not file_path.exists()

    def test_cleanup_expired_keeps_fresh_files(self, tmp_path):
        """Test cleanup_expired keeps non-expired files."""
        archive = ContentArchive(storage_path=tmp_path, ttl_hours=168, enabled=True)
        archived = archive.archive(content="Test", item_id="item-1")

        removed = archive.cleanup_expired()
        assert removed == 0

        file_path = tmp_path / f"{archived.content_hash}.json"
        assert file_path.exists()

    def test_default_ttl_is_168_hours(self):
        """Test default TTL is 7 days (168 hours)."""
        assert DEFAULT_ARCHIVE_TTL_HOURS == 168


# =============================================================================
# Test: Private Permissions
# =============================================================================


class TestContentArchivePermissions:
    """Tests for directory and file permissions."""

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Permission tests not applicable on Windows",
    )
    def test_directory_has_private_permissions(self, tmp_path):
        """Test storage directory is created with 0o700 permissions."""
        storage_path = tmp_path / "archive"
        archive = ContentArchive(storage_path=storage_path, enabled=True)
        archive.archive(content="Test", item_id="item-1")

        mode = storage_path.stat().st_mode & 0o777
        assert mode == 0o700

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Permission tests not applicable on Windows",
    )
    def test_file_has_private_permissions(self, tmp_path):
        """Test archive files are created with 0o600 permissions."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)
        archived = archive.archive(content="Test", item_id="item-1")

        file_path = tmp_path / f"{archived.content_hash}.json"
        mode = file_path.stat().st_mode & 0o777
        assert mode == 0o600


# =============================================================================
# Test: Corrupted JSON Handling
# =============================================================================


class TestContentArchiveCorruptedJSON:
    """Tests for corrupted JSON handling with skip-on-corruption policy."""

    def test_retrieve_returns_none_for_corrupt_json(self, tmp_path):
        """Test retrieve returns None for corrupted JSON file."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Create corrupt JSON file
        corrupt_hash = "a" * 64
        corrupt_file = tmp_path / f"{corrupt_hash}.json"
        corrupt_file.write_text("not valid json {{{")

        result = archive.retrieve(corrupt_hash)
        assert result is None

    def test_retrieve_logs_corruption_warning(self, tmp_path, caplog):
        """Test retrieve logs ARCHIVE_READ_CORRUPT warning."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Create corrupt JSON file
        corrupt_hash = "b" * 64
        corrupt_file = tmp_path / f"{corrupt_hash}.json"
        corrupt_file.write_text("{invalid}")

        with caplog.at_level("WARNING"):
            archive.retrieve(corrupt_hash)

        assert ARCHIVE_READ_CORRUPT in caplog.text

    def test_retrieve_by_item_id_skips_corrupt_files(self, tmp_path):
        """Test retrieve_by_item_id skips corrupted files."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Archive valid content
        archive.archive(content="Valid", item_id="target")

        # Create corrupt file
        corrupt_file = tmp_path / ("c" * 64 + ".json")
        corrupt_file.write_text("corrupt data")

        # Should only return valid record
        results = archive.retrieve_by_item_id("target")
        assert len(results) == 1
        assert results[0].content == "Valid"

    def test_archive_overwrites_corrupt_existing(self, tmp_path, caplog):
        """Test archive overwrites corrupt existing file."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Create corrupt file with known content's hash
        content = "Test content"
        content_hash = compute_content_hash(content)
        corrupt_file = tmp_path / f"{content_hash}.json"
        corrupt_file.write_text("corrupt")

        # Archive should succeed and overwrite
        with caplog.at_level("WARNING"):
            result = archive.archive(content=content, item_id="item-1")

        assert result is not None
        assert ARCHIVE_READ_CORRUPT in caplog.text

        # Verify file is now valid
        retrieved = archive.retrieve(content_hash)
        assert retrieved is not None
        assert retrieved.content == content


# =============================================================================
# Test: Read-Only Filesystem Handling
# =============================================================================


class TestContentArchiveReadOnlyFilesystem:
    """Tests for read-only filesystem graceful handling."""

    def test_archive_returns_none_on_write_failure(self, tmp_path):
        """Test archive returns None when write fails."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Mock write to fail
        with patch("tempfile.mkstemp", side_effect=OSError("Read-only")):
            result = archive.archive(content="Test", item_id="item-1")
            assert result is None

    def test_write_failure_emits_warning(self, tmp_path):
        """Test write failure adds ARCHIVE_WRITE_FAILED warning."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Mock write to fail
        with patch("tempfile.mkstemp", side_effect=OSError("Read-only")):
            archive.archive(content="Test", item_id="item-1")

        warnings = archive.warnings
        assert any(ARCHIVE_WRITE_FAILED in w for w in warnings)

    def test_write_failure_disables_archive(self, tmp_path):
        """Test write failure caches archive as not writable."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)
        assert archive.enabled is True

        # Mock write to fail
        with patch("tempfile.mkstemp", side_effect=OSError("Read-only")):
            archive.archive(content="Test", item_id="item-1")

        # Archive should now report as disabled
        assert archive.enabled is False

    def test_directory_creation_failure_handled(self, tmp_path):
        """Test directory creation failure is handled gracefully."""
        storage_path = tmp_path / "nonexistent" / "deep" / "path"

        with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
            archive = ContentArchive(storage_path=storage_path, enabled=True)
            # Should not raise, but archive should be disabled
            assert archive.enabled is False

    def test_workflow_never_blocked_by_archive_failure(self, tmp_path):
        """Test archive failures never raise exceptions."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # All these operations should return None/empty, not raise
        with patch("tempfile.mkstemp", side_effect=OSError("Failure")):
            result = archive.archive(content="Test", item_id="item-1")
            assert result is None  # No exception raised


# =============================================================================
# Test: Atomic Writes
# =============================================================================


class TestContentArchiveAtomicWrites:
    """Tests for atomic write behavior (temp file + rename)."""

    def test_archive_uses_temp_file(self, tmp_path):
        """Test archive writes to temp file before rename."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Track calls to mkstemp and rename
        temp_files_created = []
        renames_performed = []

        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            result = original_mkstemp(*args, **kwargs)
            temp_files_created.append(result[1])
            return result

        original_rename = os.rename

        def tracking_rename(src, dst):
            renames_performed.append((src, dst))
            return original_rename(src, dst)

        with (
            patch("tempfile.mkstemp", side_effect=tracking_mkstemp),
            patch("os.rename", side_effect=tracking_rename),
        ):
            archive.archive(content="Test", item_id="item-1")

        assert len(temp_files_created) == 1
        assert len(renames_performed) == 1
        # Temp file should be renamed to final path
        src, dst = renames_performed[0]
        assert src == temp_files_created[0]
        assert str(dst).endswith(".json")

    def test_temp_file_cleaned_up_on_failure(self, tmp_path):
        """Test temp file is cleaned up if write fails."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        created_temp = None

        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            nonlocal created_temp
            result = original_mkstemp(*args, **kwargs)
            created_temp = result[1]
            return result

        # Make os.rename fail after temp file is created (simulates partial failure)
        with (
            patch("tempfile.mkstemp", side_effect=tracking_mkstemp),
            patch("os.rename", side_effect=OSError("Rename failed")),
        ):
            result = archive.archive(content="Test", item_id="item-1")

        # Should have cleaned up temp file
        assert result is None
        if created_temp:
            assert not os.path.exists(created_temp)

    def test_temp_file_in_same_directory(self, tmp_path):
        """Test temp file is created in same directory as target."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        temp_dirs = []
        original_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            temp_dirs.append(kwargs.get("dir"))
            return original_mkstemp(*args, **kwargs)

        with patch("tempfile.mkstemp", side_effect=tracking_mkstemp):
            archive.archive(content="Test", item_id="item-1")

        assert len(temp_dirs) == 1
        assert temp_dirs[0] == tmp_path


# =============================================================================
# Test: Guardrails (enabled/disabled state)
# =============================================================================


class TestContentArchiveGuardrails:
    """Tests for enabled/disabled state and warnings."""

    def test_disabled_by_default(self, tmp_path):
        """Test archive is disabled by default."""
        archive = ContentArchive(storage_path=tmp_path)
        assert archive.enabled is False

    def test_disabled_archive_returns_none(self, tmp_path):
        """Test archive returns None when disabled."""
        archive = ContentArchive(storage_path=tmp_path, enabled=False)
        result = archive.archive(content="Test", item_id="item-1")
        assert result is None

    def test_enable_method(self, tmp_path):
        """Test enable() enables the archive."""
        archive = ContentArchive(storage_path=tmp_path, enabled=False)
        assert archive.enabled is False

        result = archive.enable()
        assert result is True
        assert archive.enabled is True

    def test_disable_method(self, tmp_path):
        """Test disable() disables the archive."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)
        assert archive.enabled is True

        archive.disable()
        assert archive.enabled is False

    def test_warnings_collected(self, tmp_path):
        """Test warnings are collected and retrievable."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Force a warning
        with patch("tempfile.mkstemp", side_effect=OSError("Error")):
            archive.archive(content="Test", item_id="item-1")

        warnings = archive.warnings
        assert len(warnings) > 0

    def test_clear_warnings(self, tmp_path):
        """Test clear_warnings empties the warnings list."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)

        # Force a warning
        with patch("tempfile.mkstemp", side_effect=OSError("Error")):
            archive.archive(content="Test", item_id="item-1")

        assert len(archive.warnings) > 0
        archive.clear_warnings()
        assert len(archive.warnings) == 0

    def test_get_stats_includes_state(self, tmp_path):
        """Test get_stats includes enabled/writable/warnings."""
        archive = ContentArchive(storage_path=tmp_path, enabled=True)
        archive.archive(content="Test", item_id="item-1")

        stats = archive.get_stats()
        assert "enabled" in stats
        assert "writable" in stats
        assert "warnings" in stats
        assert stats["enabled"] is True
        assert stats["count"] == 1
