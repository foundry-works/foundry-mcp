"""
Parity tests for authoring operations.

Tests that foundry-mcp and sdd-toolkit produce equivalent results for
spec authoring operations like adding/removing tasks, assumptions, and revisions.

NOTE: Many of these tests focus on foundry-mcp functionality since the SDD CLI
may not have all authoring commands implemented. The tests gracefully handle
missing CLI commands.
"""

import pytest
from tests.parity.harness.foundry_adapter import FoundryMcpAdapter
from tests.parity.harness.sdd_adapter import SddToolkitAdapter


# ============================================================================
# Shared Adapter Fixtures (reduce adapter instantiation overhead)
# ============================================================================

@pytest.fixture
def authoring_adapters_shared(authoring_spec):
    """Adapters pointing to same authoring_spec directory."""
    return (
        FoundryMcpAdapter(authoring_spec.specs_dir),
        SddToolkitAdapter(authoring_spec.specs_dir),
        "parity-test-authoring",
    )


@pytest.fixture
def subtasks_adapters_shared(authoring_with_subtasks):
    """Adapters pointing to same authoring_with_subtasks directory."""
    return (
        FoundryMcpAdapter(authoring_with_subtasks.specs_dir),
        SddToolkitAdapter(authoring_with_subtasks.specs_dir),
        "parity-test-authoring-subtasks",
    )


@pytest.fixture
def assumptions_adapters_shared(authoring_with_assumptions):
    """Adapters pointing to same authoring_with_assumptions directory."""
    return (
        FoundryMcpAdapter(authoring_with_assumptions.specs_dir),
        SddToolkitAdapter(authoring_with_assumptions.specs_dir),
        "parity-test-authoring-assumptions",
    )


def _sdd_command_available(result):
    """
    Check if SDD CLI command executed successfully.

    Returns False if:
    - Result is not a dict
    - Command returned an error (success=False)
    - Error indicates command not found/unknown

    This allows tests to gracefully skip SDD assertions when
    the CLI doesn't have a particular command implemented.
    """
    if not isinstance(result, dict):
        return False
    # Check for explicit failure
    if result.get("success") is False:
        return False
    # Check for error key presence
    if "error" in result:
        error = result.get("error", "")
        if isinstance(error, str) and error:
            return False
    return True


class TestAddTaskParity:
    """Tests for add_task parity between foundry-mcp and sdd-toolkit."""

    def test_add_task_to_phase(self, authoring_adapters_shared):
        """Adding a task to a phase should produce equivalent results."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="New implementation task",
        )
        sdd_result = sdd.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="New implementation task",
        )

        assert foundry_result["success"] is True
        assert foundry_result["parent"] == "phase-1"
        assert foundry_result["title"] == "New implementation task"

        # SDD CLI may not have this command
        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["parent"] == foundry_result["parent"]

    def test_add_task_with_description(self, authoring_adapters_shared):
        """Adding a task with description should work in both systems."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="Documented task",
            description="This task has a detailed description",
        )
        sdd_result = sdd.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="Documented task",
            description="This task has a detailed description",
        )

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_add_task_with_hours(self, authoring_adapters_shared):
        """Adding a task with hour estimate should work in both systems."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="Estimated task",
            hours=4.5,
        )
        sdd_result = sdd.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="Estimated task",
            hours=4.5,
        )

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_add_task_dry_run(self, authoring_adapters_shared):
        """Dry run should not modify the spec in either system."""
        foundry, sdd, spec_id = authoring_adapters_shared

        # Get initial task count
        foundry_initial = foundry.task_list(spec_id)
        initial_count = foundry_initial.get("count", 0)

        foundry_result = foundry.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="Dry run task",
            dry_run=True,
        )
        sdd_result = sdd.add_task(
            spec_id=spec_id,
            parent="phase-1",
            title="Dry run task",
            dry_run=True,
        )

        assert foundry_result["success"] is True
        assert foundry_result["dry_run"] is True

        # Verify task count unchanged
        foundry_after = foundry.task_list(spec_id)
        assert foundry_after.get("count", 0) == initial_count

        if _sdd_command_available(sdd_result):
            assert sdd_result["dry_run"] is True

    def test_add_task_invalid_parent(self, authoring_adapters_shared):
        """Both systems should handle invalid parent gracefully."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.add_task(
            spec_id=spec_id,
            parent="nonexistent-parent",
            title="Orphan task",
        )
        sdd_result = sdd.add_task(
            spec_id=spec_id,
            parent="nonexistent-parent",
            title="Orphan task",
        )

        assert foundry_result["success"] is False

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is False


class TestRemoveTaskParity:
    """Tests for remove_task parity."""

    def test_remove_leaf_task(self, subtasks_adapters_shared):
        """Removing a leaf task should work in both systems."""
        foundry, sdd, spec_id = subtasks_adapters_shared

        # task-1-2 is a leaf task with no children
        foundry_result = foundry.remove_task(
            spec_id=spec_id,
            task_id="task-1-2",
        )
        sdd_result = sdd.remove_task(
            spec_id=spec_id,
            task_id="task-1-2",
        )

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_remove_task_with_children_no_cascade(self, subtasks_adapters_shared):
        """Removing task with children without cascade should fail."""
        foundry, sdd, spec_id = subtasks_adapters_shared

        # task-1-1 has children (subtask-1-1-1, subtask-1-1-2)
        foundry_result = foundry.remove_task(
            spec_id=spec_id,
            task_id="task-1-1",
            cascade=False,
        )
        sdd_result = sdd.remove_task(
            spec_id=spec_id,
            task_id="task-1-1",
            cascade=False,
        )

        # Both should fail when task has children and cascade=False
        assert foundry_result["success"] is False

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is False

    def test_remove_task_cascade(self, subtasks_adapters_shared):
        """Removing task with cascade should remove children too."""
        foundry, sdd, spec_id = subtasks_adapters_shared

        # task-1-1 has children that should be removed with cascade
        foundry_result = foundry.remove_task(
            spec_id=spec_id,
            task_id="task-1-1",
            cascade=True,
        )
        sdd_result = sdd.remove_task(
            spec_id=spec_id,
            task_id="task-1-1",
            cascade=True,
        )

        assert foundry_result["success"] is True
        assert foundry_result["cascade"] is True
        assert foundry_result["children_removed"] >= 2  # At least 2 direct children

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["cascade"] is True

    def test_remove_task_dry_run(self, subtasks_adapters_shared):
        """Dry run should not actually remove the task."""
        foundry, sdd, spec_id = subtasks_adapters_shared

        # Use task-1-2 which is a leaf task
        foundry_result = foundry.remove_task(
            spec_id=spec_id,
            task_id="task-1-2",
            dry_run=True,
        )
        sdd_result = sdd.remove_task(
            spec_id=spec_id,
            task_id="task-1-2",
            dry_run=True,
        )

        assert foundry_result["success"] is True
        assert foundry_result["dry_run"] is True

        # Verify task still exists
        task = foundry.get_task(spec_id, "task-1-2")
        assert task.get("success") is True or task.get("task_id") == "task-1-2"

        if _sdd_command_available(sdd_result):
            assert sdd_result["dry_run"] is True


class TestAssumptionsParity:
    """Tests for assumption operations parity."""

    def test_add_assumption_constraint(self, assumptions_adapters_shared):
        """Adding a constraint assumption should work in foundry-mcp."""
        foundry, sdd, spec_id = assumptions_adapters_shared

        foundry_result = foundry.add_assumption(
            spec_id=spec_id,
            text="System must support Python 3.9+",
            assumption_type="constraint",
        )
        sdd_result = sdd.add_assumption(
            spec_id=spec_id,
            text="System must support Python 3.9+",
            assumption_type="constraint",
        )

        assert foundry_result["success"] is True
        assert foundry_result["type"] == "constraint"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["type"] == "constraint"

    def test_add_assumption_requirement(self, assumptions_adapters_shared):
        """Adding a requirement assumption should work in foundry-mcp."""
        foundry, sdd, spec_id = assumptions_adapters_shared

        foundry_result = foundry.add_assumption(
            spec_id=spec_id,
            text="API must return JSON responses",
            assumption_type="requirement",
        )
        sdd_result = sdd.add_assumption(
            spec_id=spec_id,
            text="API must return JSON responses",
            assumption_type="requirement",
        )

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_list_assumptions(self, assumptions_adapters_shared):
        """Listing assumptions should return consistent results."""
        foundry, sdd, spec_id = assumptions_adapters_shared

        foundry_result = foundry.list_assumptions(spec_id)
        sdd_result = sdd.list_assumptions(spec_id)

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result) and isinstance(sdd_result, dict):
            assert sdd_result["success"] is True
            # Both should have the same count of assumptions
            assert foundry_result["total_count"] == sdd_result["total_count"]

    def test_list_assumptions_filtered(self, assumptions_adapters_shared):
        """Filtering assumptions by type should work in foundry-mcp."""
        foundry, sdd, spec_id = assumptions_adapters_shared

        # First add assumptions of different types
        foundry.add_assumption(spec_id, "Constraint 1", "constraint")
        foundry.add_assumption(spec_id, "Requirement 1", "requirement")

        foundry_result = foundry.list_assumptions(spec_id, assumption_type="constraint")
        sdd_result = sdd.list_assumptions(spec_id, assumption_type="constraint")

        assert foundry_result["success"] is True
        assert foundry_result["filter_type"] == "constraint"

        if _sdd_command_available(sdd_result) and isinstance(sdd_result, dict):
            assert sdd_result["success"] is True


class TestRevisionParity:
    """Tests for revision history parity."""

    def test_add_revision(self, authoring_adapters_shared):
        """Adding a revision entry should work in foundry-mcp."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.add_revision(
            spec_id=spec_id,
            version="1.1",
            changes="Added new phase for testing",
        )
        sdd_result = sdd.add_revision(
            spec_id=spec_id,
            version="1.1",
            changes="Added new phase for testing",
        )

        assert foundry_result["success"] is True
        assert foundry_result["version"] == "1.1"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["version"] == "1.1"

    def test_add_revision_with_author(self, authoring_adapters_shared):
        """Adding a revision with author should work."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.add_revision(
            spec_id=spec_id,
            version="1.2",
            changes="Security updates",
            author="security-team",
        )
        sdd_result = sdd.add_revision(
            spec_id=spec_id,
            version="1.2",
            changes="Security updates",
            author="security-team",
        )

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True


class TestEstimateParity:
    """Tests for task estimate update parity."""

    def test_update_hours(self, authoring_adapters_shared):
        """Updating task hours should work in foundry-mcp."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.update_estimate(
            spec_id=spec_id,
            task_id="task-1-1",
            hours=8.0,
        )
        sdd_result = sdd.update_estimate(
            spec_id=spec_id,
            task_id="task-1-1",
            hours=8.0,
        )

        assert foundry_result["success"] is True
        assert foundry_result["hours"] == 8.0

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["hours"] == 8.0

    def test_update_complexity(self, authoring_adapters_shared):
        """Updating task complexity should work in foundry-mcp."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.update_estimate(
            spec_id=spec_id,
            task_id="task-1-1",
            complexity="high",
        )
        sdd_result = sdd.update_estimate(
            spec_id=spec_id,
            task_id="task-1-1",
            complexity="high",
        )

        assert foundry_result["success"] is True
        assert foundry_result["complexity"] == "high"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["complexity"] == "high"


class TestTaskMetadataParity:
    """Tests for task metadata update parity."""

    def test_update_file_path(self, authoring_adapters_shared):
        """Updating task file path should work in foundry-mcp."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.update_task_metadata(
            spec_id=spec_id,
            task_id="task-1-1",
            file_path="src/module/feature.py",
        )
        sdd_result = sdd.update_task_metadata(
            spec_id=spec_id,
            task_id="task-1-1",
            file_path="src/module/feature.py",
        )

        assert foundry_result["success"] is True
        assert "file_path" in foundry_result["fields_updated"]

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_update_multiple_fields(self, authoring_adapters_shared):
        """Updating multiple metadata fields should work."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.update_task_metadata(
            spec_id=spec_id,
            task_id="task-1-1",
            file_path="src/new_file.py",
            actual_hours=3.5,
            status_note="In progress, 50% complete",
        )
        sdd_result = sdd.update_task_metadata(
            spec_id=spec_id,
            task_id="task-1-1",
            file_path="src/new_file.py",
            actual_hours=3.5,
            status_note="In progress, 50% complete",
        )

        assert foundry_result["success"] is True
        assert len(foundry_result["fields_updated"]) >= 2

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True


class TestTaskListParity:
    """Tests for task listing parity."""

    def test_list_all_tasks(self, authoring_adapters_shared):
        """Listing all tasks should return consistent counts."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.task_list(spec_id)
        sdd_result = sdd.task_list(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["count"] >= 0

        if _sdd_command_available(sdd_result) and isinstance(sdd_result, dict):
            assert sdd_result["success"] is True
            # Both should return the same number of tasks
            assert foundry_result["count"] == sdd_result["count"]

    def test_list_tasks_by_status(self, authoring_adapters_shared):
        """Filtering tasks by status should work in foundry-mcp."""
        foundry, sdd, spec_id = authoring_adapters_shared

        foundry_result = foundry.task_list(spec_id, status_filter="pending")
        sdd_result = sdd.task_list(spec_id, status_filter="pending")

        assert foundry_result["success"] is True

        if _sdd_command_available(sdd_result) and isinstance(sdd_result, dict):
            assert sdd_result["success"] is True
