"""
Parity tests for task operations.

Tests next_task, update_status, complete_task, check_dependencies operations.
"""

import pytest

from .harness.normalizers import normalize_for_comparison
from .harness.comparators import ResultComparator


class TestNextTask:
    """Parity tests for finding next task."""

    @pytest.mark.parity
    def test_next_task_parity(self, both_adapters):
        """Test that next_task returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.next_task("parity-test-simple")
        sdd_result = sdd.next_task("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "next_task")

        # Check task_id matches
        ResultComparator.assert_key_match(
            foundry_result, sdd_result, "task_id", "next_task"
        )

    @pytest.mark.parity
    def test_next_task_no_actionable_parity(self, both_adapters):
        """Test next_task when all tasks completed."""
        foundry, sdd = both_adapters

        # Use completed spec fixture - should have no actionable tasks
        foundry_result = foundry.next_task("parity-test-completed")
        sdd_result = sdd.next_task("parity-test-completed")

        # Both should succeed but return no task
        ResultComparator.assert_success(foundry_result, sdd_result, "next_task(completed)")


class TestCheckDependencies:
    """Parity tests for dependency checking."""

    @pytest.mark.parity
    def test_check_deps_parity(self, both_adapters):
        """Test that check_dependencies returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.check_dependencies("parity-test-simple", "task-1-1")
        sdd_result = sdd.check_dependencies("parity-test-simple", "task-1-1")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "check_deps")

        # Check can_start matches
        ResultComparator.assert_key_match(
            foundry_result, sdd_result, "can_start", "check_deps"
        )

    @pytest.mark.parity
    def test_check_deps_with_blockers_parity(self, deps_spec_dir):
        """Test dependency check with actual blockers."""
        from .harness.foundry_adapter import FoundryMcpAdapter
        from .harness.sdd_adapter import SddToolkitAdapter

        specs_dir = deps_spec_dir / "specs"
        foundry = FoundryMcpAdapter(specs_dir)
        sdd = SddToolkitAdapter(specs_dir)

        # Task that depends on another
        foundry_result = foundry.check_dependencies("parity-test-deps", "task-1-2")
        sdd_result = sdd.check_dependencies("parity-test-deps", "task-1-2")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "check_deps(blocked)")


class TestUpdateStatus:
    """Parity tests for status updates."""

    @pytest.mark.parity
    def test_update_status_parity(self, isolated_adapters):
        """Test that update_status produces equivalent changes."""
        foundry, sdd = isolated_adapters

        # Update same task on both systems
        foundry_result = foundry.update_status(
            "parity-test-simple", "task-1-3", "in_progress"
        )
        sdd_result = sdd.update_status(
            "parity-test-simple", "task-1-3", "in_progress"
        )

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "update_status")

        # Verify the change took effect
        foundry_task = foundry.get_task("parity-test-simple", "task-1-3")
        sdd_task = sdd.get_task("parity-test-simple", "task-1-3")

        assert foundry_task.get("task", {}).get("status") == "in_progress"

    @pytest.mark.parity
    def test_start_task_parity(self, isolated_adapters):
        """Test that start_task produces equivalent changes."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.start_task("parity-test-simple", "task-1-3")
        sdd_result = sdd.start_task("parity-test-simple", "task-1-3")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "start_task")


class TestCompleteTask:
    """Parity tests for task completion."""

    @pytest.mark.parity
    def test_complete_task_parity(self, isolated_adapters):
        """Test that complete_task produces equivalent changes."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.complete_task(
            "parity-test-simple", "task-1-2", "Completed via parity test"
        )
        sdd_result = sdd.complete_task(
            "parity-test-simple", "task-1-2", "Completed via parity test"
        )

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "complete_task")

        # Verify the change took effect
        foundry_task = foundry.get_task("parity-test-simple", "task-1-2")
        sdd_task = sdd.get_task("parity-test-simple", "task-1-2")

        assert foundry_task.get("task", {}).get("status") == "completed"

    @pytest.mark.parity
    def test_complete_task_without_journal_parity(self, isolated_adapters):
        """Test completing task without journal entry."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.complete_task("parity-test-simple", "task-1-2")
        sdd_result = sdd.complete_task("parity-test-simple", "task-1-2")

        # Check both succeeded
        ResultComparator.assert_success(
            foundry_result, sdd_result, "complete_task(no journal)"
        )


# Standalone tests for foundry adapter only
class TestFoundryTaskOperations:
    """Tests for foundry adapter task operations."""

    def test_foundry_check_deps(self, foundry_adapter):
        """Test foundry adapter check_dependencies works."""
        result = foundry_adapter.check_dependencies("parity-test-simple", "task-1-1")
        assert result.get("success") is True
        assert "can_start" in result

    def test_foundry_update_status(self, foundry_adapter):
        """Test foundry adapter update_status works."""
        result = foundry_adapter.update_status(
            "parity-test-simple", "task-1-3", "in_progress"
        )
        assert result.get("success") is True

    def test_foundry_complete_task(self, foundry_adapter):
        """Test foundry adapter complete_task works."""
        result = foundry_adapter.complete_task(
            "parity-test-simple", "task-1-2", "Test completion"
        )
        assert result.get("success") is True
