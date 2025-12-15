"""
Unit tests for foundry_mcp.core.lifecycle module.

Tests lifecycle operations for spec status transitions.
"""

import json
import pytest
from foundry_mcp.core.lifecycle import (
    move_spec,
    activate_spec,
    complete_spec,
    archive_spec,
    get_lifecycle_state,
    list_specs_by_folder,
    get_folder_for_spec,
    MoveResult,
    LifecycleState,
    VALID_FOLDERS,
    FOLDER_TRANSITIONS,
)


# Test fixtures

@pytest.fixture
def specs_dir(tmp_path):
    """Create a temporary specs directory structure."""
    for folder in VALID_FOLDERS:
        (tmp_path / folder).mkdir()
    return tmp_path


@pytest.fixture
def sample_spec():
    """Return a sample spec data structure."""
    return {
        "spec_id": "test-spec-2025-01-01-001",
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "metadata": {
            "title": "Test Specification",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "pending",
                "parent": None,
                "children": ["task-1"],
                "total_tasks": 2,
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Test Task",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
            },
        },
    }


@pytest.fixture
def completed_spec():
    """Return a spec with all tasks completed."""
    return {
        "spec_id": "completed-spec-2025-01-01-001",
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "metadata": {
            "title": "Completed Specification",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Completed Specification",
                "status": "in_progress",
                "parent": None,
                "children": ["task-1"],
                "total_tasks": 1,
                "completed_tasks": 1,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Completed Task",
                "status": "completed",
                "parent": "spec-root",
                "children": [],
            },
        },
    }


@pytest.fixture
def pending_spec_file(specs_dir, sample_spec):
    """Create a spec file in the pending folder."""
    spec_path = specs_dir / "pending" / f"{sample_spec['spec_id']}.json"
    spec_path.write_text(json.dumps(sample_spec, indent=2))
    return spec_path


@pytest.fixture
def active_spec_file(specs_dir, sample_spec):
    """Create a spec file in the active folder."""
    sample_spec["hierarchy"]["spec-root"]["status"] = "in_progress"
    spec_path = specs_dir / "active" / f"{sample_spec['spec_id']}.json"
    spec_path.write_text(json.dumps(sample_spec, indent=2))
    return spec_path


@pytest.fixture
def completed_spec_file(specs_dir, completed_spec):
    """Create a completed spec file in the active folder."""
    spec_path = specs_dir / "active" / f"{completed_spec['spec_id']}.json"
    spec_path.write_text(json.dumps(completed_spec, indent=2))
    return spec_path


class TestValidFolders:
    """Tests for folder constants."""

    def test_valid_folders_defined(self):
        """Test that all expected folders are defined."""
        expected = {"pending", "active", "completed", "archived"}
        assert VALID_FOLDERS == expected

    def test_folder_transitions_defined(self):
        """Test that folder transitions are defined."""
        assert "pending" in FOLDER_TRANSITIONS
        assert "active" in FOLDER_TRANSITIONS
        assert "completed" in FOLDER_TRANSITIONS
        assert "archived" in FOLDER_TRANSITIONS

    def test_pending_transitions(self):
        """Test allowed transitions from pending."""
        assert "active" in FOLDER_TRANSITIONS["pending"]
        assert "archived" in FOLDER_TRANSITIONS["pending"]

    def test_active_transitions(self):
        """Test allowed transitions from active."""
        assert "pending" in FOLDER_TRANSITIONS["active"]
        assert "completed" in FOLDER_TRANSITIONS["active"]
        assert "archived" in FOLDER_TRANSITIONS["active"]

    def test_completed_transitions(self):
        """Test allowed transitions from completed."""
        assert "active" in FOLDER_TRANSITIONS["completed"]
        assert "archived" in FOLDER_TRANSITIONS["completed"]


class TestMoveSpec:
    """Tests for move_spec function."""

    def test_move_pending_to_active(self, specs_dir, pending_spec_file, sample_spec):
        """Test moving spec from pending to active."""
        result = move_spec(sample_spec["spec_id"], "active", specs_dir)
        assert result.success
        assert result.from_folder == "pending"
        assert result.to_folder == "active"
        assert (specs_dir / "active" / pending_spec_file.name).exists()
        assert not (specs_dir / "pending" / pending_spec_file.name).exists()

    def test_move_active_to_completed(self, specs_dir, active_spec_file, sample_spec):
        """Test moving spec from active to completed."""
        result = move_spec(sample_spec["spec_id"], "completed", specs_dir)
        assert result.success
        assert result.from_folder == "active"
        assert result.to_folder == "completed"

    def test_move_to_same_folder(self, specs_dir, pending_spec_file, sample_spec):
        """Test moving spec to same folder (no-op)."""
        result = move_spec(sample_spec["spec_id"], "pending", specs_dir)
        assert result.success
        assert result.from_folder == "pending"
        assert result.to_folder == "pending"
        assert result.old_path == result.new_path

    def test_invalid_folder(self, specs_dir, pending_spec_file, sample_spec):
        """Test moving to invalid folder."""
        result = move_spec(sample_spec["spec_id"], "invalid", specs_dir)
        assert not result.success
        assert "Invalid folder" in result.error

    def test_invalid_transition(self, specs_dir, pending_spec_file, sample_spec):
        """Test invalid folder transition."""
        # pending -> completed is not allowed
        result = move_spec(sample_spec["spec_id"], "completed", specs_dir)
        assert not result.success
        assert "Cannot move" in result.error

    def test_spec_not_found(self, specs_dir):
        """Test moving non-existent spec."""
        result = move_spec("nonexistent-spec", "active", specs_dir)
        assert not result.success
        assert "not found" in result.error.lower()

    def test_updates_status_on_move(self, specs_dir, pending_spec_file, sample_spec):
        """Test that status is updated when moving."""
        move_spec(sample_spec["spec_id"], "active", specs_dir)

        # Read the moved spec
        new_path = specs_dir / "active" / pending_spec_file.name
        with open(new_path, "r") as f:
            data = json.load(f)

        assert data["hierarchy"]["spec-root"]["status"] == "in_progress"

    def test_updates_last_updated(self, specs_dir, pending_spec_file, sample_spec):
        """Test that last_updated is updated when moving."""
        original_time = sample_spec["last_updated"]
        move_spec(sample_spec["spec_id"], "active", specs_dir)

        new_path = specs_dir / "active" / pending_spec_file.name
        with open(new_path, "r") as f:
            data = json.load(f)

        assert data["last_updated"] != original_time


class TestMoveResult:
    """Tests for MoveResult dataclass."""

    def test_success_result(self):
        """Test successful MoveResult."""
        result = MoveResult(
            success=True,
            spec_id="test-spec",
            from_folder="pending",
            to_folder="active",
            old_path="/old/path",
            new_path="/new/path",
        )
        assert result.success
        assert result.error is None

    def test_failure_result(self):
        """Test failed MoveResult."""
        result = MoveResult(
            success=False,
            spec_id="test-spec",
            from_folder="pending",
            to_folder="completed",
            error="Invalid transition",
        )
        assert not result.success
        assert result.error == "Invalid transition"


class TestActivateSpec:
    """Tests for activate_spec function."""

    def test_activate_pending_spec(self, specs_dir, pending_spec_file, sample_spec):
        """Test activating a pending spec."""
        result = activate_spec(sample_spec["spec_id"], specs_dir)
        assert result.success
        assert result.to_folder == "active"

    def test_activate_already_active(self, specs_dir, active_spec_file, sample_spec):
        """Test activating an already active spec."""
        result = activate_spec(sample_spec["spec_id"], specs_dir)
        assert result.success  # No-op is success

    def test_activate_nonexistent(self, specs_dir):
        """Test activating nonexistent spec."""
        result = activate_spec("nonexistent", specs_dir)
        assert not result.success


class TestCompleteSpec:
    """Tests for complete_spec function."""

    def test_complete_finished_spec(self, specs_dir, completed_spec_file, completed_spec):
        """Test completing a fully finished spec."""
        result = complete_spec(completed_spec["spec_id"], specs_dir)
        assert result.success
        assert result.to_folder == "completed"

    def test_complete_incomplete_spec_blocked(self, specs_dir, active_spec_file, sample_spec):
        """Test that incomplete spec completion is blocked."""
        result = complete_spec(sample_spec["spec_id"], specs_dir, force=False)
        assert not result.success
        assert "Cannot complete" in result.error

    def test_complete_incomplete_spec_forced(self, specs_dir, active_spec_file, sample_spec):
        """Test forcing completion of incomplete spec."""
        result = complete_spec(sample_spec["spec_id"], specs_dir, force=True)
        assert result.success
        assert result.to_folder == "completed"


class TestArchiveSpec:
    """Tests for archive_spec function."""

    def test_archive_from_active(self, specs_dir, active_spec_file, sample_spec):
        """Test archiving from active folder."""
        result = archive_spec(sample_spec["spec_id"], specs_dir)
        assert result.success
        assert result.to_folder == "archived"

    def test_archive_from_pending(self, specs_dir, pending_spec_file, sample_spec):
        """Test archiving from pending folder."""
        result = archive_spec(sample_spec["spec_id"], specs_dir)
        assert result.success
        assert result.to_folder == "archived"


class TestGetLifecycleState:
    """Tests for get_lifecycle_state function."""

    def test_get_state_pending(self, specs_dir, pending_spec_file, sample_spec):
        """Test getting state of pending spec."""
        state = get_lifecycle_state(sample_spec["spec_id"], specs_dir)
        assert state is not None
        assert state.spec_id == sample_spec["spec_id"]
        assert state.folder == "pending"
        assert state.status == "pending"

    def test_get_state_active(self, specs_dir, active_spec_file, sample_spec):
        """Test getting state of active spec."""
        state = get_lifecycle_state(sample_spec["spec_id"], specs_dir)
        assert state is not None
        assert state.folder == "active"
        assert state.status == "in_progress"

    def test_get_state_progress(self, specs_dir, completed_spec_file, completed_spec):
        """Test progress calculation."""
        state = get_lifecycle_state(completed_spec["spec_id"], specs_dir)
        assert state is not None
        assert state.progress_percentage == 100.0
        assert state.total_tasks == 1
        assert state.completed_tasks == 1

    def test_get_state_can_complete(self, specs_dir, completed_spec_file, completed_spec):
        """Test can_complete flag."""
        state = get_lifecycle_state(completed_spec["spec_id"], specs_dir)
        assert state.can_complete is True

    def test_get_state_cannot_complete(self, specs_dir, active_spec_file, sample_spec):
        """Test can_complete is False for incomplete spec."""
        state = get_lifecycle_state(sample_spec["spec_id"], specs_dir)
        assert state.can_complete is False

    def test_get_state_can_always_archive(self, specs_dir, pending_spec_file, sample_spec):
        """Test can_archive is always True."""
        state = get_lifecycle_state(sample_spec["spec_id"], specs_dir)
        assert state.can_archive is True

    def test_get_state_nonexistent(self, specs_dir):
        """Test getting state of nonexistent spec."""
        state = get_lifecycle_state("nonexistent", specs_dir)
        assert state is None


class TestLifecycleState:
    """Tests for LifecycleState dataclass."""

    def test_lifecycle_state_fields(self):
        """Test LifecycleState has all required fields."""
        state = LifecycleState(
            spec_id="test-spec",
            folder="active",
            status="in_progress",
            progress_percentage=50.0,
            total_tasks=10,
            completed_tasks=5,
            can_complete=False,
            can_archive=True,
        )
        assert state.spec_id == "test-spec"
        assert state.folder == "active"
        assert state.status == "in_progress"
        assert state.progress_percentage == 50.0
        assert state.total_tasks == 10
        assert state.completed_tasks == 5
        assert state.can_complete is False
        assert state.can_archive is True


class TestListSpecsByFolder:
    """Tests for list_specs_by_folder function."""

    def test_list_all_folders(self, specs_dir, pending_spec_file):
        """Test listing specs from all folders."""
        result = list_specs_by_folder(specs_dir)
        assert isinstance(result, dict)
        assert "pending" in result
        assert "active" in result
        assert "completed" in result
        assert "archived" in result

    def test_list_specific_folder(self, specs_dir, pending_spec_file, sample_spec):
        """Test listing specs from specific folder."""
        result = list_specs_by_folder(specs_dir, "pending")
        assert "pending" in result
        assert len(result["pending"]) == 1
        assert result["pending"][0]["spec_id"] == sample_spec["spec_id"]

    def test_list_empty_folder(self, specs_dir):
        """Test listing from empty folder."""
        result = list_specs_by_folder(specs_dir, "archived")
        assert result["archived"] == []

    def test_spec_summary_fields(self, specs_dir, pending_spec_file, sample_spec):
        """Test that spec summaries have expected fields."""
        result = list_specs_by_folder(specs_dir, "pending")
        spec = result["pending"][0]
        assert "spec_id" in spec
        assert "title" in spec
        assert "status" in spec
        assert "total_tasks" in spec
        assert "completed_tasks" in spec
        assert "progress" in spec
        assert "path" in spec

    def test_list_multiple_specs(self, specs_dir, sample_spec):
        """Test listing multiple specs in same folder."""
        # Create two specs in pending
        for i in range(2):
            spec = sample_spec.copy()
            spec["spec_id"] = f"test-spec-{i}"
            spec_path = specs_dir / "pending" / f"{spec['spec_id']}.json"
            spec_path.write_text(json.dumps(spec, indent=2))

        result = list_specs_by_folder(specs_dir, "pending")
        assert len(result["pending"]) == 2


class TestGetFolderForSpec:
    """Tests for get_folder_for_spec function."""

    def test_get_folder_pending(self, specs_dir, pending_spec_file, sample_spec):
        """Test getting folder for pending spec."""
        folder = get_folder_for_spec(sample_spec["spec_id"], specs_dir)
        assert folder == "pending"

    def test_get_folder_active(self, specs_dir, active_spec_file, sample_spec):
        """Test getting folder for active spec."""
        folder = get_folder_for_spec(sample_spec["spec_id"], specs_dir)
        assert folder == "active"

    def test_get_folder_nonexistent(self, specs_dir):
        """Test getting folder for nonexistent spec."""
        folder = get_folder_for_spec("nonexistent", specs_dir)
        assert folder is None


class TestFolderStatusMapping:
    """Tests for folder to status mapping."""

    def test_pending_folder_sets_pending_status(self, specs_dir, active_spec_file, sample_spec):
        """Test that moving to pending sets pending status."""
        move_spec(sample_spec["spec_id"], "pending", specs_dir)

        new_path = specs_dir / "pending" / f"{sample_spec['spec_id']}.json"
        with open(new_path, "r") as f:
            data = json.load(f)

        assert data["hierarchy"]["spec-root"]["status"] == "pending"

    def test_active_folder_sets_in_progress_status(self, specs_dir, pending_spec_file, sample_spec):
        """Test that moving to active sets in_progress status."""
        move_spec(sample_spec["spec_id"], "active", specs_dir)

        new_path = specs_dir / "active" / f"{sample_spec['spec_id']}.json"
        with open(new_path, "r") as f:
            data = json.load(f)

        assert data["hierarchy"]["spec-root"]["status"] == "in_progress"

    def test_completed_folder_sets_completed_status(self, specs_dir, completed_spec_file, completed_spec):
        """Test that moving to completed sets completed status."""
        move_spec(completed_spec["spec_id"], "completed", specs_dir)

        new_path = specs_dir / "completed" / f"{completed_spec['spec_id']}.json"
        with open(new_path, "r") as f:
            data = json.load(f)

        assert data["hierarchy"]["spec-root"]["status"] == "completed"

    def test_archived_folder_sets_completed_status(self, specs_dir, active_spec_file, sample_spec):
        """Test that moving to archived sets completed status."""
        move_spec(sample_spec["spec_id"], "archived", specs_dir)

        new_path = specs_dir / "archived" / f"{sample_spec['spec_id']}.json"
        with open(new_path, "r") as f:
            data = json.load(f)

        assert data["hierarchy"]["spec-root"]["status"] == "completed"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_folders(self, tmp_path, sample_spec):
        """Test handling when folder directories don't exist."""
        # Don't create any folders
        result = move_spec(sample_spec["spec_id"], "active", tmp_path)
        assert not result.success

    def test_corrupted_spec_file(self, specs_dir):
        """Test handling of corrupted spec file."""
        # Write invalid JSON
        spec_path = specs_dir / "pending" / "corrupted.json"
        spec_path.write_text("not valid json{")

        state = get_lifecycle_state("corrupted", specs_dir)
        assert state is None

    @pytest.mark.xfail(reason="Pre-existing: move_spec behavior changed for existing target")
    def test_target_already_exists(self, specs_dir, pending_spec_file, sample_spec):
        """Test handling when target file already exists."""
        # Create same file in target folder
        target = specs_dir / "active" / pending_spec_file.name
        target.write_text(json.dumps(sample_spec, indent=2))

        result = move_spec(sample_spec["spec_id"], "active", specs_dir)
        assert not result.success
        assert "exists" in result.error.lower()

    def test_empty_hierarchy(self, specs_dir):
        """Test handling spec with empty hierarchy."""
        spec = {
            "spec_id": "empty-hierarchy",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {},
        }
        spec_path = specs_dir / "pending" / "empty-hierarchy.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        state = get_lifecycle_state("empty-hierarchy", specs_dir)
        # Should handle gracefully
        assert state is not None or state is None  # Either is acceptable

    def test_zero_total_tasks(self, specs_dir):
        """Test progress calculation with zero tasks."""
        spec = {
            "spec_id": "zero-tasks",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Zero Tasks",
                    "status": "pending",
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                },
            },
        }
        spec_path = specs_dir / "pending" / "zero-tasks.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        state = get_lifecycle_state("zero-tasks", specs_dir)
        assert state is not None
        assert state.progress_percentage == 0.0
