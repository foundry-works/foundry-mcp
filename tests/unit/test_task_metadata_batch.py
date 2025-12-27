"""
Unit tests for task metadata-batch handler.

Tests batch metadata updates with flexible AND-based filtering.
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
        "spec_id": "batch-test-spec-001",
        "title": "Test Spec",
        "metadata": {"title": "Test Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["phase-1", "phase-2"]},
            "phase-1": {"type": "phase", "title": "Phase 1", "status": "pending", "parent": "spec-root", "children": ["task-1-1", "task-1-2", "verify-1-1"]},
            "task-1-1": {"type": "task", "title": "Implement feature", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "task-1-2": {"type": "task", "title": "Add tests", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "verify-1-1": {"type": "verify", "title": "Run tests", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "phase-2": {"type": "phase", "title": "Phase 2", "status": "pending", "parent": "spec-root", "children": ["task-2-1", "verify-2-1"]},
            "task-2-1": {"type": "task", "title": "Deploy feature", "status": "pending", "parent": "phase-2", "children": [], "metadata": {}},
            "verify-2-1": {"type": "verify", "title": "Smoke test", "status": "pending", "parent": "phase-2", "children": [], "metadata": {}},
        },
        "assumptions": [], "revision_history": [], "journal": [],
    }
    (specs_dir / "active" / "batch-test-spec-001.json").write_text(json.dumps(sample_spec))
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    return ServerConfig(server_name="test", server_version="0.1.0", specs_dir=test_specs_dir, log_level="WARNING")


@pytest.fixture
def task_tool(test_config):
    return create_server(test_config)._tool_manager._tools["task"].fn


# =============================================================================
# Required Parameter Tests
# =============================================================================

class TestRequiredParams:
    def test_missing_spec_id(self, task_tool):
        result = task_tool(action="metadata-batch", status_filter="pending", file_path="test.py")
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_filter(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", file_path="test.py")
        assert result["success"] is False
        assert "filter" in result["error"].lower()

    def test_missing_metadata(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending")
        assert result["success"] is False
        assert "metadata" in result["error"].lower()


# =============================================================================
# Filter Validation Tests
# =============================================================================

class TestFilterValidation:
    def test_invalid_status_filter(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="invalid", file_path="test.py")
        assert result["success"] is False
        assert "status_filter" in result["error"].lower()

    def test_invalid_pattern_regex(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="[invalid", file_path="test.py")
        assert result["success"] is False
        assert "pattern" in result["error"].lower()

    def test_empty_parent_filter(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", parent_filter="", file_path="test.py")
        assert result["success"] is False
        assert "parent_filter" in result["error"].lower()


# =============================================================================
# Metadata Validation Tests
# =============================================================================

class TestMetadataValidation:
    def test_invalid_owners_not_list(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", owners="not a list")
        assert result["success"] is False
        assert "owners" in result["error"].lower()

    def test_invalid_labels_not_dict(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", labels="not a dict")
        assert result["success"] is False
        assert "labels" in result["error"].lower()

    def test_invalid_estimated_hours(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", estimated_hours="bad")
        assert result["success"] is False
        assert "estimated_hours" in result["error"].lower()

    def test_invalid_update_metadata_not_dict(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", update_metadata="not a dict")
        assert result["success"] is False
        assert "update_metadata" in result["error"].lower()


# =============================================================================
# Filtering Tests
# =============================================================================

class TestFiltering:
    def test_filter_by_status(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", file_path="test.py", dry_run=True)
        assert result["success"] is True
        # All tasks and verify nodes are pending
        assert result["data"]["matched_count"] >= 5

    def test_filter_by_parent_filter(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", parent_filter="phase-1", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 3  # task-1-1, task-1-2, verify-1-1

    def test_filter_by_pattern(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="test", file_path="test.py", dry_run=True)
        assert result["success"] is True
        # "Add tests", "Run tests", "Smoke test" should all match
        assert result["data"]["matched_count"] >= 3

    def test_filter_combined_parent_and_pattern(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", parent_filter="phase-1", pattern="test", file_path="test.py", dry_run=True)
        assert result["success"] is True
        # "Add tests", "Run tests" in phase-1
        assert result["data"]["matched_count"] == 2

    def test_filter_no_matches(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="nonexistent", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 0

    def test_filter_legacy_phase_id(self, task_tool):
        """phase_id is a legacy alias for parent_filter."""
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", phase_id="phase-2", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 2  # task-2-1, verify-2-1


# =============================================================================
# Dry Run Tests
# =============================================================================

class TestDryRun:
    def test_dry_run_does_not_persist(self, task_tool, test_specs_dir):
        # Run with dry_run=True
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", file_path="updated.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True

        # Verify file was not modified
        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["task-1-1"]["metadata"].get("file_path") is None

    def test_dry_run_shows_preview(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", status_filter="pending", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert "nodes" in result["data"]
        assert result["data"]["matched_count"] > 0


# =============================================================================
# Persistence Tests
# =============================================================================

class TestPersistence:
    def test_update_persists(self, task_tool, test_specs_dir):
        # Run without dry_run
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", parent_filter="phase-1", file_path="updated.py", dry_run=False)
        assert result["success"] is True
        assert result["data"]["updated_count"] == 3

        # Verify file was modified
        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["task-1-1"]["metadata"].get("file_path") == "updated.py"
        assert spec_data["hierarchy"]["task-1-2"]["metadata"].get("file_path") == "updated.py"
        assert spec_data["hierarchy"]["verify-1-1"]["metadata"].get("file_path") == "updated.py"

    def test_update_multiple_fields(self, task_tool, test_specs_dir):
        result = task_tool(
            action="metadata-batch",
            spec_id="batch-test-spec-001",
            parent_filter="phase-2",
            file_path="multi.py",
            category="backend",
            estimated_hours=2.5,
            dry_run=False
        )
        assert result["success"] is True

        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        task = spec_data["hierarchy"]["task-2-1"]["metadata"]
        assert task.get("file_path") == "multi.py"
        assert task.get("category") == "backend"
        assert task.get("estimated_hours") == 2.5

    def test_update_with_custom_metadata(self, task_tool, test_specs_dir):
        result = task_tool(
            action="metadata-batch",
            spec_id="batch-test-spec-001",
            pattern="Run tests",
            update_metadata={"verification_type": "run-tests", "command": "pytest"},
            dry_run=False
        )
        assert result["success"] is True

        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        task = spec_data["hierarchy"]["verify-1-1"]["metadata"]
        assert task.get("verification_type") == "run-tests"
        assert task.get("command") == "pytest"
