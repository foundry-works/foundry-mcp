"""Tests for update_phase_metadata function."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.spec import (
    load_spec,
    update_phase_metadata,
)


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()

        # Create status directories
        (specs_dir / "pending").mkdir(parents=True)
        (specs_dir / "active").mkdir(parents=True)
        (specs_dir / "completed").mkdir(parents=True)
        (specs_dir / "archived").mkdir(parents=True)

        yield specs_dir


def _create_spec_with_phase(
    temp_specs_dir,
    spec_id: str = "test-phase-metadata",
    phase_metadata: dict = None,
) -> Path:
    """Helper to create a spec with a phase for testing."""
    if phase_metadata is None:
        phase_metadata = {"purpose": "Initial purpose", "estimated_hours": 2}

    hierarchy = {
        "spec-root": {
            "type": "spec",
            "title": "Test Spec",
            "status": "pending",
            "parent": None,
            "children": ["phase-1"],
            "total_tasks": 0,
            "completed_tasks": 0,
            "metadata": {"purpose": "", "category": "implementation"},
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        },
        "phase-1": {
            "type": "phase",
            "title": "Phase 1",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-1-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": phase_metadata,
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        },
        "task-1-1": {
            "type": "task",
            "title": "Task 1",
            "status": "pending",
            "parent": "phase-1",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {"description": "A task"},
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        },
    }

    spec_data = {
        "spec_id": spec_id,
        "title": "Test Spec",
        "metadata": {"title": "Test Spec", "version": "1.0.0"},
        "hierarchy": hierarchy,
    }

    spec_path = temp_specs_dir / "active" / f"{spec_id}.json"
    spec_path.write_text(json.dumps(spec_data))
    return spec_path


class TestUpdatePhaseMetadata:
    """Tests for update_phase_metadata function."""

    def test_update_single_field_estimated_hours(self, temp_specs_dir):
        """Should update estimated_hours successfully."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-single-field")

        result, error = update_phase_metadata(
            spec_id="test-single-field",
            phase_id="phase-1",
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["spec_id"] == "test-single-field"
        assert result["phase_id"] == "phase-1"
        assert len(result["updates"]) == 1
        assert result["updates"][0]["field"] == "estimated_hours"
        assert result["updates"][0]["new_value"] == 5.0
        assert result["updates"][0]["previous_value"] == 2

        # Verify persisted
        spec = load_spec("test-single-field", temp_specs_dir)
        assert spec["hierarchy"]["phase-1"]["metadata"]["estimated_hours"] == 5.0

    def test_update_multi_field(self, temp_specs_dir):
        """Should update multiple fields in one call."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-multi-field")

        result, error = update_phase_metadata(
            spec_id="test-multi-field",
            phase_id="phase-1",
            estimated_hours=10.0,
            description="New description",
            purpose="New purpose",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert len(result["updates"]) == 3

        fields_updated = {u["field"] for u in result["updates"]}
        assert fields_updated == {"estimated_hours", "description", "purpose"}

        # Verify persisted
        spec = load_spec("test-multi-field", temp_specs_dir)
        phase_meta = spec["hierarchy"]["phase-1"]["metadata"]
        assert phase_meta["estimated_hours"] == 10.0
        assert phase_meta["description"] == "New description"
        assert phase_meta["purpose"] == "New purpose"

    def test_dry_run_mode(self, temp_specs_dir):
        """Should return preview without saving when dry_run=True."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-dry-run")

        result, error = update_phase_metadata(
            spec_id="test-dry-run",
            phase_id="phase-1",
            estimated_hours=99.0,
            dry_run=True,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["dry_run"] is True
        assert "message" in result
        assert "Dry run" in result["message"]

        # Verify NOT persisted
        spec = load_spec("test-dry-run", temp_specs_dir)
        assert spec["hierarchy"]["phase-1"]["metadata"]["estimated_hours"] == 2

    def test_previous_value_tracking(self, temp_specs_dir):
        """Should track previous values for each updated field."""
        _create_spec_with_phase(
            temp_specs_dir,
            spec_id="test-previous-value",
            phase_metadata={"purpose": "Old purpose", "estimated_hours": 5},
        )

        result, error = update_phase_metadata(
            spec_id="test-previous-value",
            phase_id="phase-1",
            purpose="New purpose",
            estimated_hours=10.0,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        updates_by_field = {u["field"]: u for u in result["updates"]}

        assert updates_by_field["purpose"]["previous_value"] == "Old purpose"
        assert updates_by_field["purpose"]["new_value"] == "New purpose"
        assert updates_by_field["estimated_hours"]["previous_value"] == 5
        assert updates_by_field["estimated_hours"]["new_value"] == 10.0

    def test_error_missing_spec_id(self, temp_specs_dir):
        """Should return error when spec_id is empty."""
        result, error = update_phase_metadata(
            spec_id="",
            phase_id="phase-1",
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "Specification ID is required" in error

    def test_error_missing_phase_id(self, temp_specs_dir):
        """Should return error when phase_id is empty."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-missing-phase-id")

        result, error = update_phase_metadata(
            spec_id="test-missing-phase-id",
            phase_id="",
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "Phase ID is required" in error

    def test_error_no_metadata_fields(self, temp_specs_dir):
        """Should return error when no metadata fields provided."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-no-fields")

        result, error = update_phase_metadata(
            spec_id="test-no-fields",
            phase_id="phase-1",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "At least one field" in error

    def test_error_phase_not_found(self, temp_specs_dir):
        """Should return error when phase doesn't exist."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-phase-not-found")

        result, error = update_phase_metadata(
            spec_id="test-phase-not-found",
            phase_id="phase-999",
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_error_node_not_a_phase(self, temp_specs_dir):
        """Should return error when node is not a phase."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-not-a-phase")

        result, error = update_phase_metadata(
            spec_id="test-not-a-phase",
            phase_id="task-1-1",  # This is a task, not a phase
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not a phase" in error.lower()

    def test_error_negative_estimated_hours(self, temp_specs_dir):
        """Should return error when estimated_hours is negative."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-negative-hours")

        result, error = update_phase_metadata(
            spec_id="test-negative-hours",
            phase_id="phase-1",
            estimated_hours=-5.0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "must be >= 0" in error or "non-negative" in error.lower()

    def test_error_spec_not_found(self, temp_specs_dir):
        """Should return error when spec doesn't exist."""
        result, error = update_phase_metadata(
            spec_id="nonexistent-spec",
            phase_id="phase-1",
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_update_with_zero_hours(self, temp_specs_dir):
        """Should allow setting estimated_hours to zero."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-zero-hours")

        result, error = update_phase_metadata(
            spec_id="test-zero-hours",
            phase_id="phase-1",
            estimated_hours=0.0,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

        spec = load_spec("test-zero-hours", temp_specs_dir)
        assert spec["hierarchy"]["phase-1"]["metadata"]["estimated_hours"] == 0.0

    def test_update_strips_whitespace(self, temp_specs_dir):
        """Should strip whitespace from string fields."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-whitespace")

        result, error = update_phase_metadata(
            spec_id="test-whitespace",
            phase_id="phase-1",
            description="  trimmed description  ",
            purpose="  trimmed purpose  ",
            specs_dir=temp_specs_dir,
        )

        assert error is None

        spec = load_spec("test-whitespace", temp_specs_dir)
        phase_meta = spec["hierarchy"]["phase-1"]["metadata"]
        assert phase_meta["description"] == "trimmed description"
        assert phase_meta["purpose"] == "trimmed purpose"

    def test_creates_metadata_if_missing(self, temp_specs_dir):
        """Should create metadata dict if phase has none."""
        # Create spec with phase that has no metadata key
        hierarchy = {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": ["phase-1"],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                # No metadata key
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        }
        spec_data = {
            "spec_id": "test-no-metadata",
            "title": "Test Spec",
            "metadata": {"title": "Test Spec"},
            "hierarchy": hierarchy,
        }
        spec_path = temp_specs_dir / "active" / "test-no-metadata.json"
        spec_path.write_text(json.dumps(spec_data))

        result, error = update_phase_metadata(
            spec_id="test-no-metadata",
            phase_id="phase-1",
            estimated_hours=3.0,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

        spec = load_spec("test-no-metadata", temp_specs_dir)
        assert spec["hierarchy"]["phase-1"]["metadata"]["estimated_hours"] == 3.0

    def test_response_contains_phase_title(self, temp_specs_dir):
        """Should include phase_title in response."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-phase-title")

        result, error = update_phase_metadata(
            spec_id="test-phase-title",
            phase_id="phase-1",
            estimated_hours=5.0,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["phase_title"] == "Phase 1"
