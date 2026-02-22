"""Tests for update_phase_metadata function."""

import json
from pathlib import Path

from foundry_mcp.core.spec import (
    load_spec,
    update_phase_metadata,
)


def _create_spec_with_phase(
    temp_specs_dir,
    spec_id: str = "test-phase-metadata",
    phase_metadata: dict = None,
) -> Path:
    """Helper to create a spec with a phase for testing."""
    if phase_metadata is None:
        phase_metadata = {"purpose": "Initial purpose", "description": "Initial description"}

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

    def test_update_single_field_description(self, temp_specs_dir):
        """Should update description successfully."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-single-field")

        result, error = update_phase_metadata(
            spec_id="test-single-field",
            phase_id="phase-1",
            description="New description",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["spec_id"] == "test-single-field"
        assert result["phase_id"] == "phase-1"
        assert len(result["updates"]) == 1
        assert result["updates"][0]["field"] == "description"
        assert result["updates"][0]["new_value"] == "New description"

        # Verify persisted
        spec = load_spec("test-single-field", temp_specs_dir)
        assert spec["hierarchy"]["phase-1"]["metadata"]["description"] == "New description"

    def test_update_multi_field(self, temp_specs_dir):
        """Should update multiple fields in one call."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-multi-field")

        result, error = update_phase_metadata(
            spec_id="test-multi-field",
            phase_id="phase-1",
            description="New description",
            purpose="New purpose",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert len(result["updates"]) == 2

        fields_updated = {u["field"] for u in result["updates"]}
        assert fields_updated == {"description", "purpose"}

        # Verify persisted
        spec = load_spec("test-multi-field", temp_specs_dir)
        phase_meta = spec["hierarchy"]["phase-1"]["metadata"]
        assert phase_meta["description"] == "New description"
        assert phase_meta["purpose"] == "New purpose"

    def test_dry_run_mode(self, temp_specs_dir):
        """Should return preview without saving when dry_run=True."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-dry-run")

        result, error = update_phase_metadata(
            spec_id="test-dry-run",
            phase_id="phase-1",
            description="New description",
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
        assert spec["hierarchy"]["phase-1"]["metadata"]["description"] == "Initial description"

    def test_previous_value_tracking(self, temp_specs_dir):
        """Should track previous values for each updated field."""
        _create_spec_with_phase(
            temp_specs_dir,
            spec_id="test-previous-value",
            phase_metadata={"purpose": "Old purpose", "description": "Old description"},
        )

        result, error = update_phase_metadata(
            spec_id="test-previous-value",
            phase_id="phase-1",
            purpose="New purpose",
            description="New description",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        updates_by_field = {u["field"]: u for u in result["updates"]}

        assert updates_by_field["purpose"]["previous_value"] == "Old purpose"
        assert updates_by_field["purpose"]["new_value"] == "New purpose"
        assert updates_by_field["description"]["previous_value"] == "Old description"
        assert updates_by_field["description"]["new_value"] == "New description"

    def test_error_missing_spec_id(self, temp_specs_dir):
        """Should return error when spec_id is empty."""
        result, error = update_phase_metadata(
            spec_id="",
            phase_id="phase-1",
            description="test",
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
            description="test",
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
            description="test",
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
            description="test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not a phase" in error.lower()

    def test_error_spec_not_found(self, temp_specs_dir):
        """Should return error when spec doesn't exist."""
        result, error = update_phase_metadata(
            spec_id="nonexistent-spec",
            phase_id="phase-1",
            description="test",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

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
            description="Added description",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

        spec = load_spec("test-no-metadata", temp_specs_dir)
        assert spec["hierarchy"]["phase-1"]["metadata"]["description"] == "Added description"

    def test_response_contains_phase_title(self, temp_specs_dir):
        """Should include phase_title in response."""
        _create_spec_with_phase(temp_specs_dir, spec_id="test-phase-title")

        result, error = update_phase_metadata(
            spec_id="test-phase-title",
            phase_id="phase-1",
            purpose="Updated purpose",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["phase_title"] == "Phase 1"
