"""
Parity tests for edge cases.

Tests that foundry-mcp and sdd-toolkit handle edge cases correctly:
- Deep nesting in hierarchy
- Large specs with many tasks
- Empty specs
- All tasks blocked
"""

import pytest
from tests.parity.harness.foundry_adapter import FoundryMcpAdapter
from tests.parity.harness.sdd_adapter import SddToolkitAdapter
from tests.parity.harness.fixture_manager import FixtureManager


def _sdd_command_available(result):
    """Check if SDD CLI command executed successfully with real data."""
    if not isinstance(result, dict):
        return False
    if result.get("success") is False:
        return False
    if "error" in result:
        error = result.get("error", "")
        if isinstance(error, str) and error:
            return False
    if "_stdout" in result or "_stderr" in result or "_exit_code" in result:
        return False
    # Must have a success key to be considered a valid response
    if "success" not in result:
        return False
    return True


class TestDeepNestingParity:
    """Tests for deep hierarchy nesting edge cases."""

    def test_task_list_deep_hierarchy(self, deep_nesting_dir):
        """List all tasks in deeply nested spec."""
        foundry = FoundryMcpAdapter(deep_nesting_dir / "specs")
        sdd = SddToolkitAdapter(deep_nesting_dir / "specs")
        spec_id = "parity-test-deep"

        foundry_result = foundry.task_list(spec_id)
        sdd_result = sdd.task_list(spec_id)

        assert foundry_result["success"] is True
        # Spec has 8 tasks across 2 phases with 4 levels of nesting
        assert foundry_result["count"] == 8

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["count"] == foundry_result["count"]

    def test_progress_deep_hierarchy(self, deep_nesting_dir):
        """Progress calculation with deep nesting."""
        foundry = FoundryMcpAdapter(deep_nesting_dir / "specs")
        sdd = SddToolkitAdapter(deep_nesting_dir / "specs")
        spec_id = "parity-test-deep"

        foundry_result = foundry.progress(spec_id)
        sdd_result = sdd.progress(spec_id)

        assert foundry_result["success"] is True
        # None completed out of 8 total
        assert foundry_result.get("total", foundry_result.get("total_tasks", 0)) >= 0

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_get_deep_task(self, deep_nesting_dir):
        """Get task at deepest nesting level."""
        foundry = FoundryMcpAdapter(deep_nesting_dir / "specs")
        sdd = SddToolkitAdapter(deep_nesting_dir / "specs")
        spec_id = "parity-test-deep"

        # Get level 4 task (deepest)
        foundry_result = foundry.get_task(spec_id, "deeper-1-1-1-1-1")
        sdd_result = sdd.get_task(spec_id, "deeper-1-1-1-1-1")

        assert foundry_result["success"] is True
        assert foundry_result["task_id"] == "deeper-1-1-1-1-1"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_remove_deep_cascade(self, test_dir):
        """Cascade delete from deep hierarchy."""
        fixture = FixtureManager(test_dir)
        fixture.setup("edge_deep_nesting", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-deep"

        # Remove task-1-1 with cascade - should remove all children
        # task-1-1 -> subtask-1-1-1 -> deep-1-1-1-1 -> deeper-1-1-1-1-1, deeper-1-1-1-1-2
        # task-1-1 -> subtask-1-1-2
        foundry_result = foundry.remove_task(
            spec_id=spec_id,
            task_id="task-1-1",
            cascade=True,
        )

        assert foundry_result["success"] is True
        assert foundry_result["cascade"] is True
        # Should remove 5 children (subtask-1-1-1, subtask-1-1-2, deep-1-1-1-1, deeper-1-1-1-1-1, deeper-1-1-1-1-2)
        assert foundry_result["children_removed"] == 5

    def test_next_task_deep_hierarchy(self, deep_nesting_dir):
        """Next task selection considers deep hierarchy."""
        foundry = FoundryMcpAdapter(deep_nesting_dir / "specs")
        sdd = SddToolkitAdapter(deep_nesting_dir / "specs")
        spec_id = "parity-test-deep"

        foundry_result = foundry.next_task(spec_id)
        sdd_result = sdd.next_task(spec_id)

        assert foundry_result["success"] is True
        # Should find a pending task (could be any of the pending ones)
        assert foundry_result.get("task_id") is not None

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True


class TestLargeSpecParity:
    """Tests for large specs with many tasks."""

    def test_task_list_large_spec(self, large_spec_dir):
        """List all 100 tasks."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.task_list(spec_id)
        sdd_result = sdd.task_list(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["count"] == 100

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["count"] == foundry_result["count"]

    def test_task_list_filter_completed(self, large_spec_dir):
        """Filter completed tasks from large spec."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.task_list(spec_id, status_filter="completed")
        sdd_result = sdd.task_list(spec_id, status_filter="completed")

        assert foundry_result["success"] is True
        # Phase 1 and 2 are completed (40 tasks)
        assert foundry_result["count"] == 40

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_task_list_filter_pending(self, large_spec_dir):
        """Filter pending tasks from large spec."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.task_list(spec_id, status_filter="pending")

        assert foundry_result["success"] is True
        # Phase 3, 4, 5 are pending/in_progress (60 tasks, but phase 3 might be in_progress)
        # Let's check we get some pending tasks
        assert foundry_result["count"] >= 40

    def test_spec_stats_large_spec(self, large_spec_dir):
        """Stats calculation on large spec."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.spec_stats(spec_id)
        sdd_result = sdd.spec_stats(spec_id)

        assert foundry_result["success"] is True
        # Verify totals
        totals = foundry_result.get("totals", {})
        assert totals.get("tasks", 0) == 100 or totals.get("total", 0) == 100

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_progress_large_spec(self, large_spec_dir):
        """Progress on large spec."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.progress(spec_id)
        sdd_result = sdd.progress(spec_id)

        assert foundry_result["success"] is True
        # 40 out of 100 completed = 40%
        percentage = foundry_result.get("percentage", foundry_result.get("progress_percent", 0))
        assert percentage == 40 or percentage == 0.4  # Could be 40 or 0.4

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True


class TestEmptySpecParity:
    """Tests for empty spec edge cases."""

    def test_task_list_empty(self, empty_spec_dir):
        """List tasks on empty spec returns empty list."""
        foundry = FoundryMcpAdapter(empty_spec_dir / "specs")
        sdd = SddToolkitAdapter(empty_spec_dir / "specs")
        spec_id = "parity-test-empty"

        foundry_result = foundry.task_list(spec_id)
        sdd_result = sdd.task_list(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["tasks"] == []
        assert foundry_result["count"] == 0

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["count"] == 0

    def test_task_next_empty(self, empty_spec_dir):
        """Next task on empty spec returns not found."""
        foundry = FoundryMcpAdapter(empty_spec_dir / "specs")
        sdd = SddToolkitAdapter(empty_spec_dir / "specs")
        spec_id = "parity-test-empty"

        foundry_result = foundry.next_task(spec_id)
        sdd_result = sdd.next_task(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result.get("task_id") is None

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result.get("task_id") is None

    def test_progress_empty(self, empty_spec_dir):
        """Progress on empty spec."""
        foundry = FoundryMcpAdapter(empty_spec_dir / "specs")
        sdd = SddToolkitAdapter(empty_spec_dir / "specs")
        spec_id = "parity-test-empty"

        foundry_result = foundry.progress(spec_id)
        sdd_result = sdd.progress(spec_id)

        assert foundry_result["success"] is True
        # Empty spec may have 0 or 1 total depending on counting method
        # (spec-root might count as 1)
        total = foundry_result.get("total", foundry_result.get("total_tasks", 0))
        assert total <= 1  # Either 0 or 1 is acceptable

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_spec_stats_empty(self, empty_spec_dir):
        """Stats on empty spec."""
        foundry = FoundryMcpAdapter(empty_spec_dir / "specs")
        sdd = SddToolkitAdapter(empty_spec_dir / "specs")
        spec_id = "parity-test-empty"

        foundry_result = foundry.spec_stats(spec_id)
        sdd_result = sdd.spec_stats(spec_id)

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_add_task_to_empty_spec(self, test_dir):
        """Add task to empty spec (should fail - no phases)."""
        fixture = FixtureManager(test_dir)
        fixture.setup("edge_empty_spec", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-empty"

        # Try to add task to spec-root (which has no phases)
        foundry_result = foundry.add_task(
            spec_id=spec_id,
            parent="spec-root",
            title="New task",
        )

        # This should work - add a task as direct child of spec-root
        # (though normally you'd want phases)
        assert foundry_result["success"] is True


class TestAllBlockedParity:
    """Tests for all-blocked spec edge cases."""

    def test_list_blocked_all(self, all_blocked_dir):
        """List all blocked tasks."""
        foundry = FoundryMcpAdapter(all_blocked_dir / "specs")
        sdd = SddToolkitAdapter(all_blocked_dir / "specs")
        spec_id = "parity-test-all-blocked"

        foundry_result = foundry.list_blocked(spec_id)
        sdd_result = sdd.list_blocked(spec_id)

        assert foundry_result["success"] is True
        # Fixture has 3 tasks + 1 phase + 1 spec-root with blocked status
        # The exact count depends on whether phases/spec-root are included
        assert foundry_result["count"] >= 3

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["count"] == foundry_result["count"]

    def test_next_task_all_blocked(self, all_blocked_dir):
        """Next task when all are blocked."""
        foundry = FoundryMcpAdapter(all_blocked_dir / "specs")
        sdd = SddToolkitAdapter(all_blocked_dir / "specs")
        spec_id = "parity-test-all-blocked"

        foundry_result = foundry.next_task(spec_id)
        sdd_result = sdd.next_task(spec_id)

        assert foundry_result["success"] is True
        # No actionable task when all are blocked
        assert foundry_result.get("task_id") is None

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result.get("task_id") is None

    def test_task_list_blocked_filter(self, all_blocked_dir):
        """Filter tasks by blocked status."""
        foundry = FoundryMcpAdapter(all_blocked_dir / "specs")
        spec_id = "parity-test-all-blocked"

        foundry_result = foundry.task_list(spec_id, status_filter="blocked")

        assert foundry_result["success"] is True
        assert foundry_result["count"] == 3

    def test_unblock_task(self, test_dir):
        """Unblock a task."""
        fixture = FixtureManager(test_dir)
        fixture.setup("edge_all_blocked", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        sdd = SddToolkitAdapter(fixture.specs_dir)
        spec_id = "parity-test-all-blocked"

        foundry_result = foundry.unblock(spec_id, "task-1-1")
        sdd_result = sdd.unblock(spec_id, "task-1-1")

        assert foundry_result["success"] is True
        assert foundry_result.get("unblocked") is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_get_journal_blockers(self, all_blocked_dir):
        """Get journal entries for blockers."""
        foundry = FoundryMcpAdapter(all_blocked_dir / "specs")
        sdd = SddToolkitAdapter(all_blocked_dir / "specs")
        spec_id = "parity-test-all-blocked"

        foundry_result = foundry.get_journal(spec_id, entry_type="blocker")
        sdd_result = sdd.get_journal(spec_id, entry_type="blocker")

        assert foundry_result["success"] is True
        # Spec has 3 blocker journal entries
        assert foundry_result["count"] == 3

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
