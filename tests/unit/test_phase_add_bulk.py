"""
Unit tests for phase-add-bulk handler with macro format.

Tests macro payload: {spec_id, phase: {...}, tasks: [...]}
"""

import json
import pytest
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_specs_dir(tmp_path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    for d in ["active", "pending", "completed", "archived"]:
        (specs_dir / d).mkdir()

    sample_spec = {
        "spec_id": "bulk-test-spec-001",
        "title": "Test Spec",
        "metadata": {"title": "Test Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": []},
        },
        "assumptions": [], "revision_history": [], "journal": [],
    }
    (specs_dir / "active" / "bulk-test-spec-001.json").write_text(json.dumps(sample_spec))
    return specs_dir

@pytest.fixture
def test_config(test_specs_dir):
    return ServerConfig(server_name="test", server_version="0.1.0", specs_dir=test_specs_dir, log_level="WARNING")

@pytest.fixture
def authoring_tool(test_config):
    return create_server(test_config)._tool_manager._tools["authoring"].fn


# =============================================================================
# Required Parameter Tests
# =============================================================================

class TestRequiredParams:
    def test_missing_spec_id(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_phase_object(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "phase" in result["error"].lower()

    def test_phase_not_dict(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase="not a dict", tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "phase" in result["error"].lower()

    def test_missing_phase_title(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "phase.title" in result["error"].lower()

    def test_empty_phase_title(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": ""}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "phase.title" in result["error"].lower()

    def test_missing_tasks(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"})
        assert result["success"] is False
        assert "tasks" in result["error"].lower()

    def test_empty_tasks_array(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[])
        assert result["success"] is False
        assert "task" in result["error"].lower()


# =============================================================================
# Task Validation Tests
# =============================================================================

class TestTaskValidation:
    def test_task_not_dict(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=["not dict"])
        assert result["success"] is False
        assert "tasks[0]" in result["error"]

    def test_task_invalid_type(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "bad", "title": "T"}])
        assert result["success"] is False
        assert "type" in result["error"].lower()

    def test_task_missing_title(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task"}])
        assert result["success"] is False
        assert "title" in result["error"].lower()

    def test_task_estimated_hours_invalid(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T", "estimated_hours": "bad"}])
        assert result["success"] is False
        assert "estimated_hours" in result["error"]

    def test_task_estimated_hours_negative(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T", "estimated_hours": -1}])
        assert result["success"] is False
        assert "non-negative" in result["error"].lower()


# =============================================================================
# Phase Metadata Validation Tests
# =============================================================================

class TestPhaseMetadataValidation:
    def test_description_not_string(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "description": 123}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "description" in result["error"].lower()

    def test_purpose_not_string(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "purpose": 123}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "purpose" in result["error"].lower()

    def test_estimated_hours_invalid(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "estimated_hours": "bad"}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "estimated_hours" in result["error"]

    def test_metadata_defaults_not_dict(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "metadata_defaults": "bad"}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "metadata_defaults" in result["error"]

    def test_metadata_defaults_estimated_hours_invalid_string(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "metadata_defaults": {"estimated_hours": "invalid"}}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "metadata_defaults.estimated_hours" in result["error"]

    def test_metadata_defaults_estimated_hours_negative(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "metadata_defaults": {"estimated_hours": -5}}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "metadata_defaults.estimated_hours" in result["error"]

    def test_metadata_defaults_category_not_string(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "metadata_defaults": {"category": 123}}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "metadata_defaults.task_category" in result["error"]

    def test_metadata_defaults_valid_values(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P", "metadata_defaults": {"estimated_hours": 2.5, "category": "implementation"}}, tasks=[{"type": "task", "title": "T"}], dry_run=True)
        assert result["success"] is True


# =============================================================================
# Operation Parameter Tests
# =============================================================================

class TestOperationParams:
    def test_position_invalid(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}], position="bad")
        assert result["success"] is False
        assert "position" in result["error"].lower()

    def test_link_previous_invalid(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}], link_previous="bad")
        assert result["success"] is False
        assert "link_previous" in result["error"]

    def test_dry_run_invalid(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}], dry_run="bad")
        assert result["success"] is False
        assert "dry_run" in result["error"]


# =============================================================================
# Verify Task Tests
# =============================================================================

class TestVerifyTasks:
    def test_verify_type_accepted(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "verify", "title": "Run tests"}], dry_run=True)
        assert result["success"] is True
        assert result["data"]["tasks_created"][0]["type"] == "verify"

    def test_mixed_task_and_verify(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}, {"type": "verify", "title": "V"}], dry_run=True)
        assert result["success"] is True
        assert result["data"]["total_tasks"] == 2


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    def test_spec_not_found(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="nonexistent", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# =============================================================================
# Dry Run Tests
# =============================================================================

class TestDryRun:
    def test_dry_run_preview(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "Preview"}, tasks=[{"type": "task", "title": "T1"}, {"type": "task", "title": "T2"}], dry_run=True)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["phase_id"] == "(preview)"
        assert result["data"]["total_tasks"] == 2


# =============================================================================
# Success Tests
# =============================================================================

class TestSuccess:
    def test_create_phase_with_tasks(self, authoring_tool, test_specs_dir):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "New Phase", "description": "Desc"}, tasks=[{"type": "task", "title": "Task 1"}])
        assert result["success"] is True
        assert result["data"]["dry_run"] is False
        assert "phase_id" in result["data"]

        # Verify file updated
        spec = json.loads((test_specs_dir / "active" / "bulk-test-spec-001.json").read_text())
        assert result["data"]["phase_id"] in spec["hierarchy"]

    def test_create_with_all_phase_metadata(self, authoring_tool):
        result = authoring_tool(
            action="phase-add-bulk",
            spec_id="bulk-test-spec-001",
            phase={"title": "Full", "description": "D", "purpose": "P", "estimated_hours": 5.0},
            tasks=[{"type": "task", "title": "T", "estimated_hours": 2.0}],
            position=0,
            link_previous=False,
        )
        assert result["success"] is True

    def test_response_has_telemetry(self, authoring_tool):
        result = authoring_tool(action="phase-add-bulk", spec_id="bulk-test-spec-001", phase={"title": "P"}, tasks=[{"type": "task", "title": "T"}])
        assert result["success"] is True
        assert "telemetry" in result["meta"]
