"""Tests for core task operations."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.task import (
    is_unblocked,
    is_in_current_phase,
    get_next_task,
    check_dependencies,
    get_previous_sibling,
    get_parent_context,
    get_phase_context,
    get_task_journal_summary,
    prepare_task,
    add_task,
    update_task_metadata,
    move_task,
    manage_task_dependency,
    update_task_requirements,
)
from foundry_mcp.core.spec import load_spec


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve to handle macOS /var -> /private/var symlink
        specs_dir = (Path(tmpdir) / "specs").resolve()

        # Create status directories
        (specs_dir / "pending").mkdir(parents=True)
        (specs_dir / "active").mkdir(parents=True)
        (specs_dir / "completed").mkdir(parents=True)
        (specs_dir / "archived").mkdir(parents=True)

        yield specs_dir


@pytest.fixture
def sample_spec():
    """Create a sample spec data structure with phases and tasks."""
    return {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "version": "1.0.0",
        },
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Test Spec",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "completed_tasks": 1,
                "total_tasks": 2,
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1"],
                "completed_tasks": 0,
                "total_tasks": 1,
                "dependencies": {
                    "blocked_by": ["phase-1"],
                },
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1.1",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/module1.py",
                    "completed_at": "2025-01-01T00:00:00Z",
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 1.2",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/module2.py",
                },
            },
            "task-2-1": {
                "type": "task",
                "title": "Task 2.1",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "dependencies": {
                    "blocked_by": ["task-1-2"],
                },
            },
        },
        "journal": [
            {
                "timestamp": "2025-01-01T00:00:00Z",
                "task_id": "task-1-1",
                "entry_type": "status_change",
                "title": "Task Completed",
                "content": "Completed task 1.1",
                "author": "test",
            },
        ],
    }


class TestIsUnblocked:
    """Tests for is_unblocked function."""

    def test_unblocked_task(self, sample_spec):
        """Task with no blockers should be unblocked."""
        task = sample_spec["hierarchy"]["task-1-2"]
        result = is_unblocked(sample_spec, "task-1-2", task)
        assert result is True

    def test_blocked_by_incomplete_task(self, sample_spec):
        """Task with incomplete blocker should be blocked."""
        task = sample_spec["hierarchy"]["task-2-1"]
        result = is_unblocked(sample_spec, "task-2-1", task)
        assert result is False

    def test_blocked_by_completed_task(self, sample_spec):
        """Task with completed blocker should be unblocked."""
        # Complete the blocker task
        sample_spec["hierarchy"]["task-1-2"]["status"] = "completed"
        task = sample_spec["hierarchy"]["task-2-1"]
        result = is_unblocked(sample_spec, "task-2-1", task)
        # Still blocked by phase-2's blocker (phase-1 not complete)
        assert result is False

    def test_blocked_by_phase(self, sample_spec):
        """Task in blocked phase should be blocked."""
        task = sample_spec["hierarchy"]["task-2-1"]
        result = is_unblocked(sample_spec, "task-2-1", task)
        assert result is False


class TestIsInCurrentPhase:
    """Tests for is_in_current_phase function."""

    def test_task_in_phase(self, sample_spec):
        """Task should be found in its parent phase."""
        result = is_in_current_phase(sample_spec, "task-1-1", "phase-1")
        assert result is True

    def test_task_not_in_phase(self, sample_spec):
        """Task should not be found in different phase."""
        result = is_in_current_phase(sample_spec, "task-1-1", "phase-2")
        assert result is False

    def test_nonexistent_task(self, sample_spec):
        """Nonexistent task should return False."""
        result = is_in_current_phase(sample_spec, "nonexistent", "phase-1")
        assert result is False


class TestGetNextTask:
    """Tests for get_next_task function."""

    def test_finds_pending_task(self, sample_spec):
        """Should find next pending task in in_progress phase."""
        result = get_next_task(sample_spec)
        assert result is not None
        task_id, task_data = result
        assert task_id == "task-1-2"
        assert task_data["status"] == "pending"

    def test_no_tasks_when_all_completed(self, sample_spec):
        """Should return None when all tasks completed."""
        sample_spec["hierarchy"]["task-1-2"]["status"] = "completed"
        sample_spec["hierarchy"]["task-2-1"]["status"] = "completed"
        sample_spec["hierarchy"]["phase-1"]["status"] = "completed"
        sample_spec["hierarchy"]["phase-2"]["status"] = "completed"
        result = get_next_task(sample_spec)
        assert result is None

    def test_prefers_in_progress_phase(self, sample_spec):
        """Should prefer tasks from in_progress phase over pending phase."""
        result = get_next_task(sample_spec)
        assert result is not None
        task_id, _ = result
        # Should be from phase-1 which is in_progress, not phase-2 which is pending
        assert task_id == "task-1-2"


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    def test_task_with_no_dependencies(self, sample_spec):
        """Task with no dependencies should show can_start True."""
        result = check_dependencies(sample_spec, "task-1-2")
        assert result["can_start"] is True
        assert result["blocked_by"] == []
        assert result["soft_depends"] == []

    def test_task_with_blockers(self, sample_spec):
        """Task with blockers should show them."""
        result = check_dependencies(sample_spec, "task-2-1")
        assert result["can_start"] is False
        assert len(result["blocked_by"]) == 1
        assert result["blocked_by"][0]["id"] == "task-1-2"

    def test_nonexistent_task(self, sample_spec):
        """Nonexistent task should return error."""
        result = check_dependencies(sample_spec, "nonexistent")
        assert "error" in result


class TestGetPreviousSibling:
    """Tests for get_previous_sibling function."""

    def test_has_previous_sibling(self, sample_spec):
        """Task with previous sibling should return sibling info."""
        result = get_previous_sibling(sample_spec, "task-1-2")
        assert result is not None
        assert result["id"] == "task-1-1"
        assert result["title"] == "Task 1.1"
        assert result["status"] == "completed"

    def test_first_task_no_sibling(self, sample_spec):
        """First task should have no previous sibling."""
        result = get_previous_sibling(sample_spec, "task-1-1")
        assert result is None

    def test_includes_journal_excerpt(self, sample_spec):
        """Should include journal excerpt for previous sibling."""
        result = get_previous_sibling(sample_spec, "task-1-2")
        assert result is not None
        assert result["journal_excerpt"] is not None
        assert result["journal_excerpt"]["entry_type"] == "status_change"


class TestGetParentContext:
    """Tests for get_parent_context function."""

    def test_returns_parent_info(self, sample_spec):
        """Should return parent task context."""
        result = get_parent_context(sample_spec, "task-1-1")
        assert result is not None
        assert result["id"] == "phase-1"
        assert result["title"] == "Phase 1"
        assert result["type"] == "phase"

    def test_includes_children(self, sample_spec):
        """Should include list of children."""
        result = get_parent_context(sample_spec, "task-1-1")
        assert result is not None
        assert len(result["children"]) == 2
        child_ids = [c["id"] for c in result["children"]]
        assert "task-1-1" in child_ids
        assert "task-1-2" in child_ids

    def test_includes_position_label(self, sample_spec):
        """Should include position label."""
        result = get_parent_context(sample_spec, "task-1-2")
        assert result is not None
        assert result["position_label"] == "2 of 2 children"

    def test_root_task_no_parent(self, sample_spec):
        """Root task should have no parent context."""
        result = get_parent_context(sample_spec, "spec-root")
        assert result is None


class TestGetPhaseContext:
    """Tests for get_phase_context function."""

    def test_returns_phase_info(self, sample_spec):
        """Should return phase context for task."""
        result = get_phase_context(sample_spec, "task-1-1")
        assert result is not None
        assert result["title"] == "Phase 1"
        assert result["status"] == "in_progress"

    def test_includes_progress(self, sample_spec):
        """Should include progress information."""
        result = get_phase_context(sample_spec, "task-1-1")
        assert result is not None
        assert result["completed_tasks"] == 1
        assert result["total_tasks"] == 2
        assert result["percentage"] == 50

    def test_task_not_in_phase(self, sample_spec):
        """Should return None for task not in phase."""
        result = get_phase_context(sample_spec, "spec-root")
        assert result is None


class TestGetTaskJournalSummary:
    """Tests for get_task_journal_summary function."""

    def test_returns_journal_entries(self, sample_spec):
        """Should return journal entries for task."""
        result = get_task_journal_summary(sample_spec, "task-1-1")
        assert result["entry_count"] == 1
        assert len(result["entries"]) == 1
        assert result["entries"][0]["entry_type"] == "status_change"

    def test_no_entries(self, sample_spec):
        """Should return empty for task without entries."""
        result = get_task_journal_summary(sample_spec, "task-1-2")
        assert result["entry_count"] == 0
        assert result["entries"] == []


class TestPrepareTask:
    """Tests for prepare_task function.

    Uses standard response format: {success, data, error}
    """

    def test_prepares_next_task(self, temp_specs_dir, sample_spec):
        """Should prepare next actionable task."""
        # Write spec to active directory
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = prepare_task("test-spec-001", temp_specs_dir)
        assert result["success"] is True
        data = result["data"]
        assert data["task_id"] == "task-1-2"
        assert data["task_data"] is not None
        assert data["dependencies"] is not None
        assert data["context"] is not None

    def test_prepares_specific_task(self, temp_specs_dir, sample_spec):
        """Should prepare specified task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = prepare_task("test-spec-001", temp_specs_dir, task_id="task-1-1")
        assert result["success"] is True
        assert result["data"]["task_id"] == "task-1-1"

    def test_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result = prepare_task("nonexistent", temp_specs_dir)
        assert result["success"] is False
        assert result["error"] is not None

    def test_includes_context(self, temp_specs_dir, sample_spec):
        """Should include context information."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = prepare_task("test-spec-001", temp_specs_dir, task_id="task-1-2")
        context = result["data"]["context"]
        assert context["previous_sibling"] is not None
        assert context["parent_task"] is not None
        assert context["phase"] is not None

    def test_spec_complete(self, temp_specs_dir, sample_spec):
        """Should detect when spec is complete."""
        # Mark all tasks completed
        sample_spec["hierarchy"]["task-1-1"]["status"] = "completed"
        sample_spec["hierarchy"]["task-1-2"]["status"] = "completed"
        sample_spec["hierarchy"]["task-2-1"]["status"] = "completed"
        sample_spec["hierarchy"]["phase-1"]["status"] = "completed"
        sample_spec["hierarchy"]["phase-2"]["status"] = "completed"

        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = prepare_task("test-spec-001", temp_specs_dir)
        assert result["success"] is True
        assert result["data"]["spec_complete"] is True


class TestAddTask:
    """Tests for add_task function."""

    def test_add_task_to_phase(self, temp_specs_dir, sample_spec):
        """Should add task to a phase."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="New Task",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["task_id"] == "task-1-3"
        assert result["parent"] == "phase-1"
        assert result["title"] == "New Task"
        assert result["type"] == "task"

        # Verify persisted
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert "task-1-3" in spec_data["hierarchy"]
        new_task = spec_data["hierarchy"]["task-1-3"]
        assert new_task["title"] == "New Task"
        assert new_task["status"] == "pending"
        assert new_task["parent"] == "phase-1"

    def test_add_task_to_task(self, temp_specs_dir, sample_spec):
        """Should add subtask to existing task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="task-1-1",
            title="Subtask",
            task_type="subtask",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["task_id"] == "task-1-1-1"
        assert result["type"] == "subtask"

        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert "task-1-1-1" in spec_data["hierarchy"]
        assert "task-1-1-1" in spec_data["hierarchy"]["task-1-1"]["children"]

    def test_add_verify_task(self, temp_specs_dir, sample_spec):
        """Should add verify task with correct ID prefix."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Run Tests",
            task_type="verify",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["task_id"] == "verify-1-1"
        assert result["type"] == "verify"

    def test_add_task_with_description(self, temp_specs_dir, sample_spec):
        """Should include description in metadata."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Documented Task",
            description="This is the task description",
            specs_dir=temp_specs_dir
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        new_task = spec_data["hierarchy"][result["task_id"]]
        assert new_task["metadata"]["description"] == "This is the task description"

    def test_add_task_with_estimated_hours(self, temp_specs_dir, sample_spec):
        """Should include estimated_hours in metadata."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Estimated Task",
            estimated_hours=4.5,
            specs_dir=temp_specs_dir
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        new_task = spec_data["hierarchy"][result["task_id"]]
        assert new_task["metadata"]["estimated_hours"] == 4.5

    def test_add_task_at_position(self, temp_specs_dir, sample_spec):
        """Should insert task at specified position."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Inserted Task",
            position=0,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["position"] == 0

        spec_data = load_spec("test-spec-001", temp_specs_dir)
        phase = spec_data["hierarchy"]["phase-1"]
        assert phase["children"][0] == result["task_id"]

    def test_add_task_updates_ancestor_counts(self, temp_specs_dir, sample_spec):
        """Should update total_tasks for ancestors."""
        # Add required fields to make valid spec
        sample_spec["hierarchy"]["phase-1"]["total_tasks"] = 2
        sample_spec["hierarchy"]["spec-root"]["total_tasks"] = 3
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="New Task",
            specs_dir=temp_specs_dir
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert spec_data["hierarchy"]["phase-1"]["total_tasks"] == 3
        assert spec_data["hierarchy"]["spec-root"]["total_tasks"] == 4

    def test_add_task_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = add_task(
            "nonexistent-spec",
            parent_id="phase-1",
            title="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error

    def test_add_task_parent_not_found(self, temp_specs_dir, sample_spec):
        """Should return error for nonexistent parent."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="nonexistent-parent",
            title="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error

    def test_add_task_invalid_parent_type(self, temp_specs_dir, sample_spec):
        """Should reject adding task to invalid parent type."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="spec-root",  # Root is not a valid parent
            title="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Cannot add tasks" in error

    def test_add_task_empty_title(self, temp_specs_dir, sample_spec):
        """Should reject empty title."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Title is required" in error

    def test_add_task_invalid_type(self, temp_specs_dir, sample_spec):
        """Should reject invalid task type."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Test",
            task_type="invalid",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Invalid task_type" in error

    def test_add_multiple_tasks_sequential_ids(self, temp_specs_dir, sample_spec):
        """Should generate sequential IDs for multiple tasks."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result1, _ = add_task("test-spec-001", "phase-1", "Task A", specs_dir=temp_specs_dir)
        result2, _ = add_task("test-spec-001", "phase-1", "Task B", specs_dir=temp_specs_dir)
        result3, _ = add_task("test-spec-001", "phase-1", "Task C", specs_dir=temp_specs_dir)

        assert result1["task_id"] == "task-1-3"
        assert result2["task_id"] == "task-1-4"
        assert result3["task_id"] == "task-1-5"

    def test_add_task_with_file_path(self, temp_specs_dir, sample_spec):
        """Should include file_path in metadata when adding task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Task with file",
            file_path="src/components/widget.py",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["file_path"] == "src/components/widget.py"

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        new_task = spec_data["hierarchy"][result["task_id"]]
        assert new_task["metadata"]["file_path"] == "src/components/widget.py"

    def test_add_task_file_path_strips_whitespace(self, temp_specs_dir, sample_spec):
        """Should strip whitespace from file_path."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_task(
            "test-spec-001",
            parent_id="phase-1",
            title="Task with padded path",
            file_path="  src/padded.py  ",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["file_path"] == "src/padded.py"


class TestUpdateTaskMetadata:
    """Tests for update_task_metadata function."""

    def test_update_file_path(self, temp_specs_dir, sample_spec):
        """Should update file_path via update-metadata."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            file_path="src/new_module.py",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert "file_path" in result["fields_updated"]
        assert result["previous_values"]["file_path"] == "src/module2.py"

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["metadata"]["file_path"] == "src/new_module.py"

    def test_update_file_path_to_empty(self, temp_specs_dir, sample_spec):
        """Should handle setting file_path to empty string (clears it)."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            file_path="",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert "file_path" in result["fields_updated"]

        # Verify persistence - empty string becomes None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["metadata"]["file_path"] is None

    def test_update_file_path_strips_whitespace(self, temp_specs_dir, sample_spec):
        """Should strip whitespace from file_path."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            file_path="  src/trimmed.py  ",
            specs_dir=temp_specs_dir
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["metadata"]["file_path"] == "src/trimmed.py"

    def test_update_description_only(self, temp_specs_dir, sample_spec):
        """Should update description only via update-metadata."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            description="New detailed description",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["fields_updated"] == ["description"]

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["metadata"]["description"] == "New detailed description"

    def test_update_title(self, temp_specs_dir, sample_spec):
        """Should update task title via update-metadata."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            title="Updated Title",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert "title" in result["fields_updated"]
        assert result["previous_values"]["title"] == "Task 1.2"

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["title"] == "Updated Title"

    def test_update_title_empty_rejected(self, temp_specs_dir, sample_spec):
        """Should reject empty title."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            title="",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "empty" in error.lower()

    def test_update_title_whitespace_only_rejected(self, temp_specs_dir, sample_spec):
        """Should reject whitespace-only title."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            title="   ",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "empty" in error.lower() or "whitespace" in error.lower()

    def test_update_multiple_fields(self, temp_specs_dir, sample_spec):
        """Should update multiple fields at once."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            title="New Title",
            file_path="src/new.py",
            description="New description",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert "title" in result["fields_updated"]
        assert "file_path" in result["fields_updated"]
        assert "description" in result["fields_updated"]

    def test_update_dry_run(self, temp_specs_dir, sample_spec):
        """Dry run should preview without saving."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            title="Would Be New",
            dry_run=True,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["dry_run"] is True
        assert "title" in result["fields_updated"]

        # Verify not saved
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["title"] == "Task 1.2"  # Unchanged

    def test_update_task_not_found(self, temp_specs_dir, sample_spec):
        """Should return error for nonexistent task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "nonexistent-task",
            title="New Title",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_update_no_fields_provided(self, temp_specs_dir, sample_spec):
        """Should require at least one field to update."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_metadata(
            "test-spec-001",
            "task-1-2",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "at least one field" in error.lower()


class TestMoveTask:
    """Tests for move_task function."""

    def test_reorder_within_parent(self, temp_specs_dir, sample_spec):
        """Should reorder task within same parent."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # Move task-1-2 to position 1 (first)
        result, error, warnings = move_task(
            "test-spec-001",
            "task-1-2",
            position=1,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["is_reparenting"] is False
        assert result["old_position"] == 2
        assert result["new_position"] == 1

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        phase = spec_data["hierarchy"]["phase-1"]
        assert phase["children"][0] == "task-1-2"
        assert phase["children"][1] == "task-1-1"

    def test_reparent_to_different_phase(self, temp_specs_dir, sample_spec):
        """Should move task to different phase."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error, warnings = move_task(
            "test-spec-001",
            "task-1-2",
            new_parent="phase-2",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["is_reparenting"] is True
        assert result["old_parent"] == "phase-1"
        assert result["new_parent"] == "phase-2"

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["parent"] == "phase-2"
        assert "task-1-2" in spec_data["hierarchy"]["phase-2"]["children"]
        assert "task-1-2" not in spec_data["hierarchy"]["phase-1"]["children"]

    def test_move_prevents_circular_reference(self, temp_specs_dir, sample_spec):
        """Should prevent moving task to its descendant."""
        # Add a child task to task-1-1 (using type=task so it can be a parent)
        sample_spec["hierarchy"]["task-1-1"]["children"] = ["task-1-1-1"]
        sample_spec["hierarchy"]["task-1-1"]["total_tasks"] = 2
        sample_spec["hierarchy"]["task-1-1-1"] = {
            "type": "task",
            "title": "Child Task",
            "status": "pending",
            "parent": "task-1-1",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # Try to move task-1-1 under its own child
        result, error, warnings = move_task(
            "test-spec-001",
            "task-1-1",
            new_parent="task-1-1-1",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "circular" in error.lower()

    def test_move_dry_run(self, temp_specs_dir, sample_spec):
        """Dry run should preview without saving."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error, warnings = move_task(
            "test-spec-001",
            "task-1-2",
            new_parent="phase-2",
            dry_run=True,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["dry_run"] is True
        assert result["is_reparenting"] is True

        # Verify not saved
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["parent"] == "phase-1"  # Unchanged

    def test_move_task_not_found(self, temp_specs_dir, sample_spec):
        """Should return error for nonexistent task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error, warnings = move_task(
            "test-spec-001",
            "nonexistent-task",
            new_parent="phase-2",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_move_invalid_position(self, temp_specs_dir, sample_spec):
        """Should reject invalid position."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error, warnings = move_task(
            "test-spec-001",
            "task-1-2",
            position=100,  # Way too high
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "invalid position" in error.lower() or "must be" in error.lower()

    def test_move_updates_task_counts(self, temp_specs_dir, sample_spec):
        """Should update task counts on affected parents."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # Move completed task from phase-1 to phase-2
        result, error, warnings = move_task(
            "test-spec-001",
            "task-1-1",  # This task is completed
            new_parent="phase-2",
            specs_dir=temp_specs_dir
        )

        assert error is None

        # Verify task counts updated
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        phase1 = spec_data["hierarchy"]["phase-1"]
        phase2 = spec_data["hierarchy"]["phase-2"]

        # Phase-1 should have 1 less total and 1 less completed
        assert phase1["total_tasks"] == 1  # Was 2
        assert phase1["completed_tasks"] == 0  # Was 1

        # Phase-2 should have 1 more total and 1 more completed
        assert phase2["total_tasks"] == 2  # Was 1
        assert phase2["completed_tasks"] == 1  # Was 0


class TestManageTaskDependency:
    """Tests for manage_task_dependency function."""

    def test_add_blocks_creates_reciprocal(self, temp_specs_dir, sample_spec):
        """Adding blocks should create reciprocal blocked_by on target."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["action"] == "add"
        assert result["dependency_type"] == "blocks"
        assert result["source_task"] == "task-1-1"
        assert result["target_task"] == "task-1-2"

        # Verify source has target in blocks
        assert "task-1-2" in result["source_dependencies"]["blocks"]

        # Verify target has source in blocked_by (reciprocal)
        assert "task-1-1" in result["target_dependencies"]["blocked_by"]

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        source = spec_data["hierarchy"]["task-1-1"]
        target = spec_data["hierarchy"]["task-1-2"]
        assert "task-1-2" in source["dependencies"]["blocks"]
        assert "task-1-1" in target["dependencies"]["blocked_by"]

    def test_add_blocked_by_creates_reciprocal(self, temp_specs_dir, sample_spec):
        """Adding blocked_by should create reciprocal blocks on target."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-2",
            target_task_id="task-1-1",
            dependency_type="blocked_by",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None

        # Verify source has target in blocked_by
        assert "task-1-1" in result["source_dependencies"]["blocked_by"]

        # Verify target has source in blocks (reciprocal)
        assert "task-1-2" in result["target_dependencies"]["blocks"]

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        source = spec_data["hierarchy"]["task-1-2"]
        target = spec_data["hierarchy"]["task-1-1"]
        assert "task-1-1" in source["dependencies"]["blocked_by"]
        assert "task-1-2" in target["dependencies"]["blocks"]

    def test_add_depends_no_reciprocal(self, temp_specs_dir, sample_spec):
        """Adding depends should NOT create reciprocal (soft dependency)."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-2",
            target_task_id="task-1-1",
            dependency_type="depends",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None

        # Verify source has target in depends
        assert "task-1-1" in result["source_dependencies"]["depends"]

        # Verify target does NOT have source in any dependency list (no reciprocal)
        assert "task-1-2" not in result["target_dependencies"]["depends"]
        assert "task-1-2" not in result["target_dependencies"]["blocks"]
        assert "task-1-2" not in result["target_dependencies"]["blocked_by"]

    def test_remove_blocks_removes_reciprocal(self, temp_specs_dir, sample_spec):
        """Removing blocks should also remove reciprocal blocked_by."""
        # Set up existing dependency
        sample_spec["hierarchy"]["task-1-1"]["dependencies"] = {
            "blocks": ["task-1-2"],
            "blocked_by": [],
            "depends": [],
        }
        sample_spec["hierarchy"]["task-1-2"]["dependencies"] = {
            "blocks": [],
            "blocked_by": ["task-1-1"],
            "depends": [],
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="remove",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["action"] == "remove"

        # Verify both sides are cleared
        assert "task-1-2" not in result["source_dependencies"]["blocks"]
        assert "task-1-1" not in result["target_dependencies"]["blocked_by"]

        # Verify persistence
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        source = spec_data["hierarchy"]["task-1-1"]
        target = spec_data["hierarchy"]["task-1-2"]
        assert "task-1-2" not in source["dependencies"]["blocks"]
        assert "task-1-1" not in target["dependencies"]["blocked_by"]

    def test_remove_nonexistent_dependency_error(self, temp_specs_dir, sample_spec):
        """Removing non-existent dependency should return error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="remove",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "does not exist" in error.lower()

    def test_add_duplicate_dependency_error(self, temp_specs_dir, sample_spec):
        """Adding already existing dependency should return error."""
        # Set up existing dependency
        sample_spec["hierarchy"]["task-1-1"]["dependencies"] = {
            "blocks": ["task-1-2"],
            "blocked_by": [],
            "depends": [],
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "already exists" in error.lower()

    def test_circular_dependency_self_reference(self, temp_specs_dir, sample_spec):
        """Task cannot depend on itself."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-1",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "cannot depend on itself" in error.lower()

    def test_circular_dependency_direct(self, temp_specs_dir, sample_spec):
        """Direct circular dependency should be prevented."""
        # Set up: task-1-1 blocks task-1-2
        sample_spec["hierarchy"]["task-1-1"]["dependencies"] = {
            "blocks": ["task-1-2"],
            "blocked_by": [],
            "depends": [],
        }
        sample_spec["hierarchy"]["task-1-2"]["dependencies"] = {
            "blocks": [],
            "blocked_by": ["task-1-1"],
            "depends": [],
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # Try to add: task-1-2 blocks task-1-1 (would create cycle)
        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-2",
            target_task_id="task-1-1",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "circular" in error.lower()

    def test_circular_dependency_transitive(self, temp_specs_dir, sample_spec):
        """Transitive circular dependency should be prevented."""
        # Add task-2-1 for the chain: task-1-1 -> task-1-2 -> task-2-1 -> task-1-1
        sample_spec["hierarchy"]["task-1-1"]["dependencies"] = {
            "blocks": ["task-1-2"],
            "blocked_by": [],
            "depends": [],
        }
        sample_spec["hierarchy"]["task-1-2"]["dependencies"] = {
            "blocks": ["task-2-1"],
            "blocked_by": ["task-1-1"],
            "depends": [],
        }
        sample_spec["hierarchy"]["task-2-1"]["dependencies"] = {
            "blocks": [],
            "blocked_by": ["task-1-2"],
            "depends": [],
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # Try to add: task-2-1 blocks task-1-1 (would create transitive cycle)
        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-2-1",
            target_task_id="task-1-1",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "circular" in error.lower()

    def test_source_task_not_found(self, temp_specs_dir, sample_spec):
        """Should return error for nonexistent source task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="nonexistent-task",
            target_task_id="task-1-1",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_target_task_not_found(self, temp_specs_dir, sample_spec):
        """Should return error for nonexistent target task."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="nonexistent-task",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = manage_task_dependency(
            "nonexistent-spec",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_invalid_dependency_type(self, temp_specs_dir, sample_spec):
        """Should reject invalid dependency_type."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="invalid",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "invalid" in error.lower()

    def test_invalid_action(self, temp_specs_dir, sample_spec):
        """Should reject invalid action."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="invalid",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert error is not None
        assert "invalid" in error.lower()

    def test_dry_run_add(self, temp_specs_dir, sample_spec):
        """Dry run should preview add without saving."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="add",
            dry_run=True,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["dry_run"] is True
        assert "task-1-2" in result["source_dependencies"]["blocks"]

        # Verify NOT saved
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        source = spec_data["hierarchy"]["task-1-1"]
        # Dependencies might be None or missing in original spec
        deps = source.get("dependencies", {})
        assert "task-1-2" not in deps.get("blocks", [])

    def test_dry_run_remove(self, temp_specs_dir, sample_spec):
        """Dry run should preview remove without saving."""
        # Set up existing dependency
        sample_spec["hierarchy"]["task-1-1"]["dependencies"] = {
            "blocks": ["task-1-2"],
            "blocked_by": [],
            "depends": [],
        }
        sample_spec["hierarchy"]["task-1-2"]["dependencies"] = {
            "blocks": [],
            "blocked_by": ["task-1-1"],
            "depends": [],
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="remove",
            dry_run=True,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["dry_run"] is True
        assert "task-1-2" not in result["source_dependencies"]["blocks"]

        # Verify NOT saved (dependency still exists)
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        source = spec_data["hierarchy"]["task-1-1"]
        assert "task-1-2" in source["dependencies"]["blocks"]

    def test_phase_to_phase_dependency(self, temp_specs_dir, sample_spec):
        """Should support dependencies between phases."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="phase-1",
            target_task_id="phase-2",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert "phase-2" in result["source_dependencies"]["blocks"]
        assert "phase-1" in result["target_dependencies"]["blocked_by"]

    def test_creates_dependencies_dict_if_missing(self, temp_specs_dir, sample_spec):
        """Should create dependencies dict if not present on task."""
        # Remove dependencies from tasks
        if "dependencies" in sample_spec["hierarchy"]["task-1-1"]:
            del sample_spec["hierarchy"]["task-1-1"]["dependencies"]
        if "dependencies" in sample_spec["hierarchy"]["task-1-2"]:
            del sample_spec["hierarchy"]["task-1-2"]["dependencies"]

        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = manage_task_dependency(
            "test-spec-001",
            source_task_id="task-1-1",
            target_task_id="task-1-2",
            dependency_type="blocks",
            action="add",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert "task-1-2" in result["source_dependencies"]["blocks"]

        # Verify persistence - dependencies dicts created
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        source = spec_data["hierarchy"]["task-1-1"]
        target = spec_data["hierarchy"]["task-1-2"]
        assert "dependencies" in source
        assert "dependencies" in target
        assert "task-1-2" in source["dependencies"]["blocks"]
        assert "task-1-1" in target["dependencies"]["blocked_by"]


class TestUpdateTaskRequirements:
    """Tests for update_task_requirements function."""

    def test_add_acceptance_requirement(self, temp_specs_dir, sample_spec):
        """Should add acceptance requirement with auto-generated ID."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="add",
            requirement_type="acceptance", text="User can submit form",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["requirement"]["id"] == "req-1"
        assert result["requirement"]["type"] == "acceptance"
        assert result["total_requirements"] == 1

    def test_add_technical_and_constraint_types(self, temp_specs_dir, sample_spec):
        """Should support technical and constraint requirement types."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        r1, _ = update_task_requirements(
            "test-spec-001", "task-1-2", action="add",
            requirement_type="technical", text="Use async", specs_dir=temp_specs_dir
        )
        r2, _ = update_task_requirements(
            "test-spec-001", "task-1-2", action="add",
            requirement_type="constraint", text="<200ms", specs_dir=temp_specs_dir
        )

        assert r1["requirement"]["type"] == "technical"
        assert r2["requirement"]["type"] == "constraint"
        assert r2["requirement"]["id"] == "req-2"

    def test_remove_requirement(self, temp_specs_dir, sample_spec):
        """Should remove requirement by ID."""
        sample_spec["hierarchy"]["task-1-2"]["metadata"] = {
            "requirements": [{"id": "req-1", "type": "acceptance", "text": "Test"}]
        }
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="remove",
            requirement_id="req-1", specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["action"] == "remove"
        assert result["total_requirements"] == 0

    def test_remove_nonexistent_error(self, temp_specs_dir, sample_spec):
        """Should error when removing non-existent requirement."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="remove",
            requirement_id="req-999", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error.lower()

    def test_invalid_action_error(self, temp_specs_dir, sample_spec):
        """Should reject invalid action."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="invalid", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "invalid" in error.lower()

    def test_invalid_requirement_type_error(self, temp_specs_dir, sample_spec):
        """Should reject invalid requirement type."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="add",
            requirement_type="invalid", text="Test", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "invalid" in error.lower()

    def test_empty_text_error(self, temp_specs_dir, sample_spec):
        """Should reject empty text."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="add",
            requirement_type="acceptance", text="", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "non-empty" in error.lower()

    def test_dry_run_preview(self, temp_specs_dir, sample_spec):
        """Dry run should preview without saving."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_task_requirements(
            "test-spec-001", "task-1-2", action="add",
            requirement_type="acceptance", text="Test",
            dry_run=True, specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["dry_run"] is True

        # Verify NOT saved
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        reqs = spec_data["hierarchy"]["task-1-2"].get("metadata", {}).get("requirements", [])
        assert len(reqs) == 0
