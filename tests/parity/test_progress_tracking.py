"""
Parity tests for progress tracking operations.

Tests progress, journal, and blocked task operations.
"""

import pytest

from .harness.normalizers import normalize_for_comparison
from .harness.comparators import ResultComparator


class TestProgress:
    """Parity tests for progress tracking."""

    @pytest.mark.parity
    def test_progress_parity(self, both_adapters):
        """Test that progress returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.progress("parity-test-simple")
        sdd_result = sdd.progress("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "progress")

        # Check spec_id matches
        ResultComparator.assert_key_match(
            foundry_result, sdd_result, "spec_id", "progress"
        )

    @pytest.mark.parity
    def test_progress_multi_phase_parity(self, fixture_manager):
        """Test progress for multi-phase spec."""
        from .harness.foundry_adapter import FoundryMcpAdapter
        from .harness.sdd_adapter import SddToolkitAdapter

        fixture_manager.setup("multi_phase_spec", status="active")
        specs_dir = fixture_manager.specs_dir

        foundry = FoundryMcpAdapter(specs_dir)
        sdd = SddToolkitAdapter(specs_dir)

        foundry_result = foundry.progress("parity-test-multi-phase")
        sdd_result = sdd.progress("parity-test-multi-phase")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "progress(multi-phase)")


class TestJournal:
    """Parity tests for journal operations."""

    @pytest.mark.parity
    def test_add_journal_parity(self, isolated_adapters):
        """Test that add_journal produces equivalent entries."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.add_journal(
            spec_id="parity-test-simple",
            title="Test Entry",
            content="This is a test journal entry",
            entry_type="note",
        )
        sdd_result = sdd.add_journal(
            spec_id="parity-test-simple",
            title="Test Entry",
            content="This is a test journal entry",
            entry_type="note",
        )

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "add_journal")

    @pytest.mark.parity
    def test_add_journal_with_task_parity(self, isolated_adapters):
        """Test adding journal entry associated with a task."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.add_journal(
            spec_id="parity-test-simple",
            title="Task Note",
            content="Note about task progress",
            entry_type="note",
            task_id="task-1-1",
        )
        sdd_result = sdd.add_journal(
            spec_id="parity-test-simple",
            title="Task Note",
            content="Note about task progress",
            entry_type="note",
            task_id="task-1-1",
        )

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "add_journal(task)")

    @pytest.mark.parity
    def test_get_journal_parity(self, both_adapters):
        """Test that get_journal returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.get_journal("parity-test-simple")
        sdd_result = sdd.get_journal("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "get_journal")

    @pytest.mark.parity
    def test_get_journal_by_type_parity(self, isolated_adapters):
        """Test filtering journal by entry type."""
        foundry, sdd = isolated_adapters

        # Add some entries first
        foundry.add_journal("parity-test-simple", "Note", "A note", "note")
        foundry.add_journal("parity-test-simple", "Decision", "A decision", "decision")
        sdd.add_journal("parity-test-simple", "Note", "A note", "note")
        sdd.add_journal("parity-test-simple", "Decision", "A decision", "decision")

        # Get filtered journal
        foundry_result = foundry.get_journal("parity-test-simple", entry_type="note")
        sdd_result = sdd.get_journal("parity-test-simple", entry_type="note")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "get_journal(type)")


class TestBlockedTasks:
    """Parity tests for blocked task operations."""

    @pytest.mark.parity
    def test_mark_blocked_parity(self, isolated_adapters):
        """Test that mark_blocked produces equivalent results."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.mark_blocked(
            "parity-test-simple", "task-1-3", "Waiting for external API"
        )
        sdd_result = sdd.mark_blocked(
            "parity-test-simple", "task-1-3", "Waiting for external API"
        )

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "mark_blocked")

    @pytest.mark.parity
    def test_unblock_parity(self, isolated_adapters):
        """Test that unblock produces equivalent results."""
        foundry, sdd = isolated_adapters

        # First block the task
        foundry.mark_blocked("parity-test-simple", "task-1-3", "Blocked for test")
        sdd.mark_blocked("parity-test-simple", "task-1-3", "Blocked for test")

        # Now unblock
        foundry_result = foundry.unblock("parity-test-simple", "task-1-3")
        sdd_result = sdd.unblock("parity-test-simple", "task-1-3")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "unblock")

    @pytest.mark.parity
    def test_list_blocked_parity(self, isolated_adapters):
        """Test that list_blocked returns equivalent results."""
        foundry, sdd = isolated_adapters

        # Block a task first
        foundry.mark_blocked("parity-test-simple", "task-1-3", "Test blocker")
        sdd.mark_blocked("parity-test-simple", "task-1-3", "Test blocker")

        # List blocked
        foundry_result = foundry.list_blocked("parity-test-simple")
        sdd_result = sdd.list_blocked("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "list_blocked")

    @pytest.mark.parity
    def test_list_blocked_empty_parity(self, both_adapters):
        """Test list_blocked when no tasks are blocked."""
        foundry, sdd = both_adapters

        foundry_result = foundry.list_blocked("parity-test-simple")
        sdd_result = sdd.list_blocked("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "list_blocked(empty)")


# Standalone tests for foundry adapter
class TestFoundryProgressOperations:
    """Tests for foundry adapter progress operations."""

    def test_foundry_progress(self, foundry_adapter):
        """Test foundry adapter progress works."""
        result = foundry_adapter.progress("parity-test-simple")
        assert result.get("success") is True
        assert result.get("spec_id") == "parity-test-simple"

    def test_foundry_add_journal(self, foundry_adapter):
        """Test foundry adapter add_journal works."""
        result = foundry_adapter.add_journal(
            "parity-test-simple", "Test", "Test content", "note"
        )
        assert result.get("success") is True

    def test_foundry_get_journal(self, foundry_adapter):
        """Test foundry adapter get_journal works."""
        result = foundry_adapter.get_journal("parity-test-simple")
        assert result.get("success") is True
        assert "entries" in result

    def test_foundry_mark_blocked(self, foundry_adapter):
        """Test foundry adapter mark_blocked works."""
        result = foundry_adapter.mark_blocked(
            "parity-test-simple", "task-1-3", "Test reason"
        )
        assert result.get("success") is True

    def test_foundry_list_blocked(self, foundry_adapter):
        """Test foundry adapter list_blocked works."""
        result = foundry_adapter.list_blocked("parity-test-simple")
        assert result.get("success") is True
