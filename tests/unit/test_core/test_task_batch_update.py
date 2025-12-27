"""Tests for batch_update_tasks function."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.task import batch_update_tasks
from foundry_mcp.core.spec import load_spec


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()
        (specs_dir / "pending").mkdir(parents=True)
        (specs_dir / "active").mkdir(parents=True)
        (specs_dir / "completed").mkdir(parents=True)
        (specs_dir / "archived").mkdir(parents=True)
        yield specs_dir


@pytest.fixture
def batch_spec():
    """Create a spec with multiple tasks for batch testing."""
    return {
        "spec_id": "batch-test-001",
        "title": "Batch Test Spec",
        "metadata": {"title": "Batch Test Spec"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Batch Test",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1", "task-2-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Implement feature A",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Old description"},
            },
            "task-1-2": {
                "type": "task",
                "title": "Implement feature B",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {},
            },
            "task-1-3": {
                "type": "task",
                "title": "Fix bug in feature A",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"category": "bugfix"},
            },
            "task-2-1": {
                "type": "task",
                "title": "Implement feature C",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {},
            },
            "task-2-2": {
                "type": "verify",
                "title": "Run tests",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {},
            },
        },
        "journal": [],
    }


class TestBatchUpdateByStatusFilter:
    """Tests for filtering by status."""

    def test_filter_by_pending_status(self, temp_specs_dir, batch_spec):
        """Should update only pending tasks."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            description="Updated via batch",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["matched_count"] == 4  # task-1-2, task-1-3, task-2-1, task-2-2
        assert result["updated_count"] == 4

    def test_filter_by_completed_status(self, temp_specs_dir, batch_spec):
        """Should update only completed tasks."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="completed",
            description="Completed task updated",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["matched_count"] == 1  # Only task-1-1
        assert result["nodes"][0]["node_id"] == "task-1-1"

    def test_invalid_status_filter(self, temp_specs_dir, batch_spec):
        """Should reject invalid status filter."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="invalid_status",
            description="Test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "Invalid status_filter" in error


class TestBatchUpdateByParentFilter:
    """Tests for filtering by parent."""

    def test_filter_by_phase(self, temp_specs_dir, batch_spec):
        """Should update only tasks under specified phase."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            parent_filter="phase-1",
            description="Phase 1 tasks",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["matched_count"] == 3  # task-1-1, task-1-2, task-1-3
        node_ids = [n["node_id"] for n in result["nodes"]]
        assert "task-1-1" in node_ids
        assert "task-1-2" in node_ids
        assert "task-1-3" in node_ids
        assert "task-2-1" not in node_ids

    def test_parent_not_found(self, temp_specs_dir, batch_spec):
        """Should error for nonexistent parent."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            parent_filter="phase-nonexistent",
            description="Test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not found" in error.lower()


class TestBatchUpdateByPattern:
    """Tests for filtering by regex pattern."""

    def test_filter_by_title_pattern(self, temp_specs_dir, batch_spec):
        """Should match tasks by title pattern."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            pattern="feature",  # Matches titles containing "feature"
            description="Feature task",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # task-1-1, task-1-2, task-1-3 ("Fix bug in feature A"), task-2-1
        assert result["matched_count"] == 4

    def test_filter_by_id_pattern(self, temp_specs_dir, batch_spec):
        """Should match tasks by ID pattern."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            pattern="task-1-",  # Matches task-1-1, task-1-2, task-1-3
            description="Phase 1 task",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["matched_count"] == 3

    def test_case_insensitive_pattern(self, temp_specs_dir, batch_spec):
        """Pattern matching should be case-insensitive."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            pattern="FEATURE",  # Should still match (case-insensitive)
            description="Feature task",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # Same 4 tasks as test_filter_by_title_pattern
        assert result["matched_count"] == 4

    def test_invalid_regex_pattern(self, temp_specs_dir, batch_spec):
        """Should reject invalid regex pattern."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            pattern="[invalid",  # Invalid regex
            description="Test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "Invalid regex" in error


class TestBatchUpdateCombinedFilters:
    """Tests for combined filters with AND logic."""

    def test_status_and_parent_filter(self, temp_specs_dir, batch_spec):
        """Should apply both status and parent filters."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            parent_filter="phase-1",
            description="Pending in phase 1",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["matched_count"] == 2  # task-1-2, task-1-3 (not task-1-1 which is completed)

    def test_status_and_pattern_filter(self, temp_specs_dir, batch_spec):
        """Should apply both status and pattern filters."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            pattern="feature",
            description="Pending feature tasks",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # task-1-2 (pending, "feature B"), task-1-3 (pending, "Fix bug in feature A"),
        # task-2-1 (pending, "feature C")
        # NOT task-1-1 (completed)
        assert result["matched_count"] == 3


class TestBatchUpdateMetadataFields:
    """Tests for updating different metadata fields."""

    def test_update_description(self, temp_specs_dir, batch_spec):
        """Should update description field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="completed",
            description="New description",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # Verify change was applied
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-1"]
        assert task["metadata"]["description"] == "New description"

    def test_update_file_path(self, temp_specs_dir, batch_spec):
        """Should update file_path field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            pattern="task-1-",
            file_path="src/features/module.py",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        for task_id in ["task-1-1", "task-1-2", "task-1-3"]:
            task = spec_data["hierarchy"][task_id]
            assert task["metadata"]["file_path"] == "src/features/module.py"

    def test_update_estimated_hours(self, temp_specs_dir, batch_spec):
        """Should update estimated_hours field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            parent_filter="phase-2",
            estimated_hours=2.5,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        assert spec_data["hierarchy"]["task-2-1"]["metadata"]["estimated_hours"] == 2.5

    def test_update_labels(self, temp_specs_dir, batch_spec):
        """Should update labels field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            labels={"priority": "high", "team": "backend"},
            specs_dir=temp_specs_dir,
        )

        assert error is None
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["metadata"]["labels"] == {"priority": "high", "team": "backend"}

    def test_update_owners(self, temp_specs_dir, batch_spec):
        """Should update owners field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            parent_filter="phase-1",
            owners=["alice", "bob"],
            specs_dir=temp_specs_dir,
        )

        assert error is None
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        for task_id in ["task-1-1", "task-1-2", "task-1-3"]:
            task = spec_data["hierarchy"][task_id]
            assert task["metadata"]["owners"] == ["alice", "bob"]

    def test_update_category(self, temp_specs_dir, batch_spec):
        """Should update category field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            pattern="feature",
            category="feature",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["metadata"]["category"] == "feature"


class TestBatchUpdateDryRun:
    """Tests for dry_run mode."""

    def test_dry_run_no_changes(self, temp_specs_dir, batch_spec):
        """Dry run should not modify spec."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            description="Should not be saved",
            dry_run=True,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["dry_run"] is True
        assert result["matched_count"] == 4
        assert result["updated_count"] == 0

        # Verify no changes persisted
        spec_data = load_spec("batch-test-001", temp_specs_dir)
        task = spec_data["hierarchy"]["task-1-2"]
        assert task["metadata"].get("description") is None

    def test_dry_run_shows_diff(self, temp_specs_dir, batch_spec):
        """Dry run should show old/new values in diff."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="completed",
            description="New desc",
            dry_run=True,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # task-1-1 has old description
        node = result["nodes"][0]
        assert "diff" in node
        assert node["diff"]["description"]["old"] == "Old description"
        assert node["diff"]["description"]["new"] == "New desc"


class TestBatchUpdateValidation:
    """Tests for validation errors."""

    def test_no_filter_provided(self, temp_specs_dir, batch_spec):
        """Should require at least one filter."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            description="Test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "at least one filter" in error.lower()

    def test_no_metadata_provided(self, temp_specs_dir, batch_spec):
        """Should require at least one metadata field."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "at least one metadata" in error.lower()

    def test_spec_not_found(self, temp_specs_dir):
        """Should error for nonexistent spec."""
        result, error = batch_update_tasks(
            "nonexistent-spec",
            status_filter="pending",
            description="Test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not found" in error.lower()

    def test_no_matches(self, temp_specs_dir, batch_spec):
        """Should return empty result when no tasks match."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="blocked",  # No blocked tasks
            description="Test",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["matched_count"] == 0
        assert result["message"] == "No tasks matched"


class TestBatchUpdateMaxMatches:
    """Tests for max_matches safety limit."""

    def test_respects_max_matches(self, temp_specs_dir, batch_spec):
        """Should limit matches to max_matches."""
        spec_file = temp_specs_dir / "active" / "batch-test-001.json"
        spec_file.write_text(json.dumps(batch_spec))

        result, error = batch_update_tasks(
            "batch-test-001",
            status_filter="pending",
            description="Limited update",
            max_matches=2,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["matched_count"] == 2
        assert result["skipped_count"] == 2
        assert "warnings" in result
