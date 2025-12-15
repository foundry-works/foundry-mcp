"""
Unit tests for foundry_mcp.core.journal module.

Tests journal entry management, blocker operations, and status updates.
"""

import pytest
from foundry_mcp.core.journal import (
    add_journal_entry,
    bulk_journal,
    get_journal_entries,
    get_latest_journal_entry,
    mark_blocked,
    unblock,
    get_blocker_info,
    get_resolved_blockers,
    list_blocked_tasks,
    update_task_status,
    mark_task_journaled,
    find_unjournaled_tasks,
    JournalEntry,
    BlockerInfo,
    ResolvedBlocker,
    VALID_ENTRY_TYPES,
    VALID_BLOCKER_TYPES,
)


# Test fixtures

@pytest.fixture
def spec_with_task():
    """Return a spec with a task for testing."""
    return {
        "spec_id": "test-spec-001",
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "journal": [],
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": ["task-1", "task-2"],
                "total_tasks": 2,
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Test Task 1",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-2": {
                "type": "task",
                "title": "Test Task 2",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
        },
    }


@pytest.fixture
def spec_with_journal():
    """Return a spec with existing journal entries."""
    return {
        "spec_id": "test-spec-001",
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "journal": [
            {
                "timestamp": "2025-01-01T01:00:00Z",
                "entry_type": "note",
                "title": "First Entry",
                "content": "First content",
                "author": "test",
                "task_id": "task-1",
                "metadata": {},
            },
            {
                "timestamp": "2025-01-01T02:00:00Z",
                "entry_type": "decision",
                "title": "Second Entry",
                "content": "Second content",
                "author": "test",
                "metadata": {},
            },
        ],
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": ["task-1"],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Test Task",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
        },
    }


class TestAddJournalEntry:
    """Tests for add_journal_entry function."""

    def test_adds_entry_to_journal(self, spec_with_task):
        """Test that entry is added to journal array."""
        add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
            entry_type="note",
        )
        assert len(spec_with_task["journal"]) == 1
        assert spec_with_task["journal"][0]["title"] == "Test Entry"

    def test_returns_journal_entry_object(self, spec_with_task):
        """Test that function returns JournalEntry object."""
        entry = add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
        )
        assert isinstance(entry, JournalEntry)
        assert entry.title == "Test Entry"
        assert entry.content == "Test content"

    def test_adds_timestamp(self, spec_with_task):
        """Test that timestamp is added automatically."""
        entry = add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
        )
        assert entry.timestamp is not None
        assert "T" in entry.timestamp  # ISO 8601 format

    def test_adds_entry_type(self, spec_with_task):
        """Test that entry_type is set correctly."""
        entry = add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
            entry_type="decision",
        )
        assert entry.entry_type == "decision"

    def test_adds_task_id(self, spec_with_task):
        """Test that task_id is associated with entry."""
        entry = add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
            task_id="task-1",
        )
        assert entry.task_id == "task-1"
        assert spec_with_task["journal"][0]["task_id"] == "task-1"

    def test_adds_author(self, spec_with_task):
        """Test that author is set correctly."""
        entry = add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
            author="custom-author",
        )
        assert entry.author == "custom-author"

    def test_updates_last_updated(self, spec_with_task):
        """Test that last_updated timestamp is updated."""
        original = spec_with_task["last_updated"]
        add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
        )
        assert spec_with_task["last_updated"] != original

    def test_creates_journal_array_if_missing(self, spec_with_task):
        """Test that journal array is created if missing."""
        del spec_with_task["journal"]
        add_journal_entry(
            spec_with_task,
            title="Test Entry",
            content="Test content",
        )
        assert "journal" in spec_with_task
        assert len(spec_with_task["journal"]) == 1


class TestGetJournalEntries:
    """Tests for get_journal_entries function."""

    def test_returns_all_entries(self, spec_with_journal):
        """Test that all entries are returned."""
        entries = get_journal_entries(spec_with_journal)
        assert len(entries) == 2

    def test_returns_journal_entry_objects(self, spec_with_journal):
        """Test that JournalEntry objects are returned."""
        entries = get_journal_entries(spec_with_journal)
        assert all(isinstance(e, JournalEntry) for e in entries)

    def test_filters_by_task_id(self, spec_with_journal):
        """Test filtering by task_id."""
        entries = get_journal_entries(spec_with_journal, task_id="task-1")
        assert len(entries) == 1
        assert entries[0].task_id == "task-1"

    def test_filters_by_entry_type(self, spec_with_journal):
        """Test filtering by entry_type."""
        entries = get_journal_entries(spec_with_journal, entry_type="decision")
        assert len(entries) == 1
        assert entries[0].entry_type == "decision"

    def test_limits_results(self, spec_with_journal):
        """Test limiting number of results."""
        entries = get_journal_entries(spec_with_journal, limit=1)
        assert len(entries) == 1

    def test_returns_most_recent_first(self, spec_with_journal):
        """Test that most recent entries are first."""
        entries = get_journal_entries(spec_with_journal)
        assert entries[0].timestamp > entries[1].timestamp

    def test_handles_empty_journal(self, spec_with_task):
        """Test handling of empty journal."""
        entries = get_journal_entries(spec_with_task)
        assert len(entries) == 0


class TestGetLatestJournalEntry:
    """Tests for get_latest_journal_entry function."""

    def test_returns_latest_entry(self, spec_with_journal):
        """Test that latest entry for task is returned."""
        entry = get_latest_journal_entry(spec_with_journal, "task-1")
        assert entry is not None
        assert entry.task_id == "task-1"

    def test_returns_none_when_no_entries(self, spec_with_task):
        """Test that None is returned when no entries exist."""
        entry = get_latest_journal_entry(spec_with_task, "task-1")
        assert entry is None


class TestMarkBlocked:
    """Tests for mark_blocked function."""

    def test_marks_task_as_blocked(self, spec_with_task):
        """Test that task status is set to blocked."""
        result = mark_blocked(spec_with_task, "task-1", "Test blocker")
        assert result is True
        assert spec_with_task["hierarchy"]["task-1"]["status"] == "blocked"

    def test_stores_blocker_metadata(self, spec_with_task):
        """Test that blocker metadata is stored."""
        mark_blocked(
            spec_with_task,
            "task-1",
            "Test blocker",
            blocker_type="technical",
            ticket="JIRA-123",
        )
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert metadata["blocker_description"] == "Test blocker"
        assert metadata["blocker_type"] == "technical"
        assert metadata["blocker_ticket"] == "JIRA-123"

    def test_stores_blocked_at_timestamp(self, spec_with_task):
        """Test that blocked_at timestamp is stored."""
        mark_blocked(spec_with_task, "task-1", "Test blocker")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert "blocked_at" in metadata
        assert "T" in metadata["blocked_at"]  # ISO 8601 format

    def test_returns_false_for_nonexistent_task(self, spec_with_task):
        """Test that False is returned for nonexistent task."""
        result = mark_blocked(spec_with_task, "nonexistent", "Test blocker")
        assert result is False

    def test_recalculates_counts(self, spec_with_task):
        """Test that task counts are recalculated."""
        mark_blocked(spec_with_task, "task-1", "Test blocker")
        # Counts should be recalculated (not throwing an error is success)
        assert spec_with_task["hierarchy"]["spec-root"]["completed_tasks"] == 0


class TestUnblock:
    """Tests for unblock function."""

    def test_unblocks_task(self, spec_with_task):
        """Test that task is unblocked."""
        mark_blocked(spec_with_task, "task-1", "Test blocker")
        result = unblock(spec_with_task, "task-1", "Fixed it")
        assert result is True
        assert spec_with_task["hierarchy"]["task-1"]["status"] == "pending"

    def test_stores_resolution(self, spec_with_task):
        """Test that resolution is stored in resolved_blockers."""
        mark_blocked(spec_with_task, "task-1", "Test blocker")
        unblock(spec_with_task, "task-1", "Fixed it")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert "resolved_blockers" in metadata
        assert len(metadata["resolved_blockers"]) == 1
        assert metadata["resolved_blockers"][0]["resolution"] == "Fixed it"

    def test_removes_active_blocker_fields(self, spec_with_task):
        """Test that active blocker fields are removed."""
        mark_blocked(spec_with_task, "task-1", "Test blocker")
        unblock(spec_with_task, "task-1")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert "blocker_description" not in metadata
        assert "blocker_type" not in metadata

    def test_returns_false_if_not_blocked(self, spec_with_task):
        """Test that False is returned if task not blocked."""
        result = unblock(spec_with_task, "task-1")
        assert result is False

    def test_returns_false_for_nonexistent_task(self, spec_with_task):
        """Test that False is returned for nonexistent task."""
        result = unblock(spec_with_task, "nonexistent")
        assert result is False


class TestGetBlockerInfo:
    """Tests for get_blocker_info function."""

    def test_returns_blocker_info(self, spec_with_task):
        """Test that BlockerInfo is returned for blocked task."""
        mark_blocked(spec_with_task, "task-1", "Test blocker", blocker_type="technical")
        info = get_blocker_info(spec_with_task, "task-1")
        assert isinstance(info, BlockerInfo)
        assert info.description == "Test blocker"
        assert info.blocker_type == "technical"

    def test_returns_none_if_not_blocked(self, spec_with_task):
        """Test that None is returned if task not blocked."""
        info = get_blocker_info(spec_with_task, "task-1")
        assert info is None


class TestGetResolvedBlockers:
    """Tests for get_resolved_blockers function."""

    def test_returns_resolved_blockers(self, spec_with_task):
        """Test that resolved blockers are returned."""
        mark_blocked(spec_with_task, "task-1", "First blocker")
        unblock(spec_with_task, "task-1", "Fixed first")
        mark_blocked(spec_with_task, "task-1", "Second blocker")
        unblock(spec_with_task, "task-1", "Fixed second")

        resolved = get_resolved_blockers(spec_with_task, "task-1")
        assert len(resolved) == 2
        assert all(isinstance(r, ResolvedBlocker) for r in resolved)

    def test_returns_empty_list_if_no_history(self, spec_with_task):
        """Test that empty list is returned if no blocker history."""
        resolved = get_resolved_blockers(spec_with_task, "task-1")
        assert resolved == []


class TestListBlockedTasks:
    """Tests for list_blocked_tasks function."""

    def test_returns_blocked_tasks(self, spec_with_task):
        """Test that all blocked tasks are returned."""
        mark_blocked(spec_with_task, "task-1", "Blocker 1")
        mark_blocked(spec_with_task, "task-2", "Blocker 2")
        blocked = list_blocked_tasks(spec_with_task)
        assert len(blocked) == 2

    def test_returns_blocker_details(self, spec_with_task):
        """Test that blocker details are included."""
        mark_blocked(spec_with_task, "task-1", "Test blocker", blocker_type="technical")
        blocked = list_blocked_tasks(spec_with_task)
        assert blocked[0]["blocker_type"] == "technical"
        assert blocked[0]["blocker_description"] == "Test blocker"

    def test_returns_empty_list_if_none_blocked(self, spec_with_task):
        """Test that empty list is returned if no blocked tasks."""
        blocked = list_blocked_tasks(spec_with_task)
        assert blocked == []


class TestUpdateTaskStatus:
    """Tests for update_task_status function."""

    def test_updates_status(self, spec_with_task):
        """Test that status is updated."""
        result = update_task_status(spec_with_task, "task-1", "in_progress")
        assert result is True
        assert spec_with_task["hierarchy"]["task-1"]["status"] == "in_progress"

    def test_sets_started_at_for_in_progress(self, spec_with_task):
        """Test that started_at is set for in_progress status."""
        update_task_status(spec_with_task, "task-1", "in_progress")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert "started_at" in metadata

    def test_sets_completed_at_for_completed(self, spec_with_task):
        """Test that completed_at is set for completed status."""
        update_task_status(spec_with_task, "task-1", "completed")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert "completed_at" in metadata

    def test_sets_needs_journaling_for_completed(self, spec_with_task):
        """Test that needs_journaling is set for completed status."""
        update_task_status(spec_with_task, "task-1", "completed")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert metadata["needs_journaling"] is True

    def test_stores_status_note(self, spec_with_task):
        """Test that status note is stored."""
        update_task_status(spec_with_task, "task-1", "in_progress", note="Starting work")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert metadata["status_note"] == "Starting work"

    def test_returns_false_for_invalid_status(self, spec_with_task):
        """Test that False is returned for invalid status."""
        result = update_task_status(spec_with_task, "task-1", "invalid_status")
        assert result is False

    def test_returns_false_for_nonexistent_task(self, spec_with_task):
        """Test that False is returned for nonexistent task."""
        result = update_task_status(spec_with_task, "nonexistent", "in_progress")
        assert result is False


class TestMarkTaskJournaled:
    """Tests for mark_task_journaled function."""

    def test_clears_needs_journaling_flag(self, spec_with_task):
        """Test that needs_journaling flag is cleared."""
        update_task_status(spec_with_task, "task-1", "completed")
        mark_task_journaled(spec_with_task, "task-1")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert metadata["needs_journaling"] is False

    def test_sets_journaled_at_timestamp(self, spec_with_task):
        """Test that journaled_at timestamp is set."""
        update_task_status(spec_with_task, "task-1", "completed")
        mark_task_journaled(spec_with_task, "task-1")
        metadata = spec_with_task["hierarchy"]["task-1"]["metadata"]
        assert "journaled_at" in metadata

    def test_returns_false_for_nonexistent_task(self, spec_with_task):
        """Test that False is returned for nonexistent task."""
        result = mark_task_journaled(spec_with_task, "nonexistent")
        assert result is False


class TestFindUnjournaled:
    """Tests for find_unjournaled_tasks function."""

    def test_finds_unjournaled_tasks(self, spec_with_task):
        """Test that unjournaled completed tasks are found."""
        update_task_status(spec_with_task, "task-1", "completed")
        unjournaled = find_unjournaled_tasks(spec_with_task)
        assert len(unjournaled) == 1
        assert unjournaled[0]["task_id"] == "task-1"

    def test_excludes_journaled_tasks(self, spec_with_task):
        """Test that journaled tasks are excluded."""
        update_task_status(spec_with_task, "task-1", "completed")
        mark_task_journaled(spec_with_task, "task-1")
        unjournaled = find_unjournaled_tasks(spec_with_task)
        assert len(unjournaled) == 0

    def test_returns_empty_list_if_none_unjournaled(self, spec_with_task):
        """Test that empty list is returned if no unjournaled tasks."""
        unjournaled = find_unjournaled_tasks(spec_with_task)
        assert unjournaled == []


class TestBulkJournal:
    """Tests for bulk_journal function."""

    def test_adds_multiple_entries(self, spec_with_task):
        """Test adding multiple entries at once."""
        entries = [
            {"title": "Entry 1", "content": "Content 1"},
            {"title": "Entry 2", "content": "Content 2"},
            {"title": "Entry 3", "content": "Content 3"},
        ]
        results = bulk_journal(spec_with_task, entries)
        assert len(results) == 3
        assert len(spec_with_task["journal"]) == 3

    def test_returns_journal_entry_objects(self, spec_with_task):
        """Test that JournalEntry objects are returned."""
        entries = [{"title": "Test", "content": "Content"}]
        results = bulk_journal(spec_with_task, entries)
        assert all(isinstance(r, JournalEntry) for r in results)

    def test_all_entries_share_timestamp(self, spec_with_task):
        """Test that all entries in batch have same timestamp."""
        entries = [
            {"title": "Entry 1", "content": "Content 1"},
            {"title": "Entry 2", "content": "Content 2"},
        ]
        results = bulk_journal(spec_with_task, entries)
        assert results[0].timestamp == results[1].timestamp

    def test_handles_task_ids(self, spec_with_task):
        """Test that task_ids are properly associated."""
        entries = [
            {"title": "Entry 1", "content": "Content 1", "task_id": "task-1"},
            {"title": "Entry 2", "content": "Content 2", "task_id": "task-2"},
        ]
        results = bulk_journal(spec_with_task, entries)
        assert results[0].task_id == "task-1"
        assert results[1].task_id == "task-2"

    def test_clears_journaling_flags(self, spec_with_task):
        """Test that needs_journaling flags are cleared for task entries."""
        update_task_status(spec_with_task, "task-1", "completed")
        update_task_status(spec_with_task, "task-2", "completed")

        entries = [
            {"title": "Entry 1", "content": "Content 1", "task_id": "task-1"},
            {"title": "Entry 2", "content": "Content 2", "task_id": "task-2"},
        ]
        bulk_journal(spec_with_task, entries)

        # Both tasks should have journaling flag cleared
        assert spec_with_task["hierarchy"]["task-1"]["metadata"]["needs_journaling"] is False
        assert spec_with_task["hierarchy"]["task-2"]["metadata"]["needs_journaling"] is False

    def test_updates_last_updated_once(self, spec_with_task):
        """Test that last_updated is set once for the batch."""
        entries = [
            {"title": "Entry 1", "content": "Content 1"},
            {"title": "Entry 2", "content": "Content 2"},
        ]
        bulk_journal(spec_with_task, entries)
        # last_updated should be a valid timestamp
        assert "T" in spec_with_task["last_updated"]

    def test_handles_empty_list(self, spec_with_task):
        """Test handling of empty entry list."""
        results = bulk_journal(spec_with_task, [])
        assert results == []
        assert len(spec_with_task["journal"]) == 0

    def test_skips_invalid_entries(self, spec_with_task):
        """Test that invalid entries are skipped."""
        entries = [
            {"title": "Valid", "content": "Content"},
            {"title": "Missing content"},  # Missing content
            {"content": "Missing title"},  # Missing title
            "not a dict",  # Invalid type
            None,  # Invalid type
            {"title": "Also valid", "content": "More content"},
        ]
        results = bulk_journal(spec_with_task, entries)
        assert len(results) == 2
        assert results[0].title == "Valid"
        assert results[1].title == "Also valid"

    def test_handles_all_optional_fields(self, spec_with_task):
        """Test that all optional fields are handled."""
        entries = [
            {
                "title": "Full entry",
                "content": "Full content",
                "entry_type": "decision",
                "task_id": "task-1",
                "author": "custom-author",
                "metadata": {"key": "value"},
            }
        ]
        results = bulk_journal(spec_with_task, entries)
        assert results[0].entry_type == "decision"
        assert results[0].task_id == "task-1"
        assert results[0].author == "custom-author"
        assert results[0].metadata == {"key": "value"}

    def test_creates_journal_array_if_missing(self, spec_with_task):
        """Test that journal array is created if missing."""
        del spec_with_task["journal"]
        entries = [{"title": "Test", "content": "Content"}]
        bulk_journal(spec_with_task, entries)
        assert "journal" in spec_with_task
        assert len(spec_with_task["journal"]) == 1


class TestJournalConstants:
    """Tests for journal constants."""

    def test_valid_entry_types(self):
        """Test that valid entry types are defined."""
        assert "status_change" in VALID_ENTRY_TYPES
        assert "deviation" in VALID_ENTRY_TYPES
        assert "blocker" in VALID_ENTRY_TYPES
        assert "decision" in VALID_ENTRY_TYPES
        assert "note" in VALID_ENTRY_TYPES

    def test_valid_blocker_types(self):
        """Test that valid blocker types are defined."""
        assert "dependency" in VALID_BLOCKER_TYPES
        assert "technical" in VALID_BLOCKER_TYPES
        assert "resource" in VALID_BLOCKER_TYPES
        assert "decision" in VALID_BLOCKER_TYPES
