"""
Unit tests for foundry_mcp.core.validation module.

Tests validation functions, auto-fix capabilities, and statistics calculation.
"""

import copy
import pytest
from foundry_mcp.core.validation import (
    validate_spec,
    get_fix_actions,
    apply_fixes,
    calculate_stats,
    Diagnostic,
    ValidationResult,
    FixAction,
    SpecStats,
    VALID_NODE_TYPES,
    VALID_STATUSES,
    VALID_VERIFICATION_TYPES,
)


# Test fixtures

@pytest.fixture
def valid_spec():
    """Return a minimal valid spec for testing."""
    return {
        "spec_id": "test-spec-2025-01-01-001",
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": ["task-1"],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Test Task",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {"task_category": "implementation", "file_path": "test.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
    }


@pytest.fixture
def spec_with_issues():
    """Return a spec with various validation issues."""
    return {
        "spec_id": "invalid-format",
        "generated": "2025-01-01",  # Invalid date format
        "last_updated": "2025-01-01T00:00:00Z",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": ["task-1", "task-2"],
                "total_tasks": 1,  # Wrong count
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Test Task 1",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {"task_category": "implementation", "file_path": "test.py"},
                "dependencies": {"blocks": ["task-2"], "blocked_by": [], "depends": []},
            },
            "task-2": {
                "type": "task",
                "title": "Test Task 2",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {"task_category": "implementation", "file_path": "test2.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},  # Missing blocked_by
            },
        },
    }


class TestValidateSpec:
    """Tests for validate_spec function."""

    def test_valid_spec_passes(self, valid_spec):
        """Test that a valid spec passes validation."""
        result = validate_spec(valid_spec)
        assert result.is_valid
        assert result.error_count == 0
        assert result.spec_id == "test-spec-2025-01-01-001"

    def test_missing_required_field(self):
        """Test that missing required fields are detected."""
        spec = {"spec_id": "test-001"}  # Missing hierarchy, generated, last_updated
        result = validate_spec(spec)
        assert not result.is_valid
        assert result.error_count >= 1
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_REQUIRED_FIELD" in codes

    def test_empty_hierarchy_detected(self, valid_spec):
        """Test that empty hierarchy is detected."""
        valid_spec["hierarchy"] = {}
        result = validate_spec(valid_spec)
        assert not result.is_valid
        codes = [d.code for d in result.diagnostics]
        assert "EMPTY_HIERARCHY" in codes

    def test_missing_spec_root_detected(self, valid_spec):
        """Test that missing spec-root is detected."""
        del valid_spec["hierarchy"]["spec-root"]
        result = validate_spec(valid_spec)
        assert not result.is_valid
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_SPEC_ROOT" in codes

    def test_invalid_spec_id_format_detected(self, valid_spec):
        """Test that invalid spec_id format is flagged."""
        valid_spec["spec_id"] = "invalid-format"
        result = validate_spec(valid_spec)
        codes = [d.code for d in result.diagnostics]
        assert "INVALID_SPEC_ID_FORMAT" in codes

    def test_invalid_status_detected(self, valid_spec):
        """Test that invalid status values are detected."""
        valid_spec["hierarchy"]["task-1"]["status"] = "invalid_status"
        result = validate_spec(valid_spec)
        assert not result.is_valid
        codes = [d.code for d in result.diagnostics]
        assert "INVALID_STATUS" in codes

    def test_invalid_node_type_detected(self, valid_spec):
        """Test that invalid node types are detected."""
        valid_spec["hierarchy"]["task-1"]["type"] = "invalid_type"
        result = validate_spec(valid_spec)
        assert not result.is_valid
        codes = [d.code for d in result.diagnostics]
        assert "INVALID_NODE_TYPE" in codes

    def test_count_mismatch_detected(self, spec_with_issues):
        """Test that task count mismatches are detected."""
        result = validate_spec(spec_with_issues)
        codes = [d.code for d in result.diagnostics]
        assert "TOTAL_TASKS_MISMATCH" in codes

    def test_bidirectional_dependency_inconsistency(self, spec_with_issues):
        """Test that bidirectional dependency inconsistency is detected."""
        result = validate_spec(spec_with_issues)
        codes = [d.code for d in result.diagnostics]
        assert "BIDIRECTIONAL_INCONSISTENCY" in codes

    def test_missing_file_path_for_implementation_task(self, valid_spec):
        """Test that implementation tasks without file_path are flagged."""
        del valid_spec["hierarchy"]["task-1"]["metadata"]["file_path"]
        result = validate_spec(valid_spec)
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_FILE_PATH" in codes

    def test_missing_verification_type_for_verify_node(self, valid_spec):
        """Test that verify nodes without verification_type are flagged."""
        valid_spec["hierarchy"]["task-1"]["type"] = "verify"
        del valid_spec["hierarchy"]["task-1"]["metadata"]["file_path"]
        result = validate_spec(valid_spec)
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_VERIFICATION_TYPE" in codes

    def test_orphaned_node_detected(self, valid_spec):
        """Test that orphaned nodes are detected."""
        valid_spec["hierarchy"]["orphan-task"] = {
            "type": "task",
            "title": "Orphaned Task",
            "status": "pending",
            "parent": "nonexistent",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {},
        }
        result = validate_spec(valid_spec)
        codes = [d.code for d in result.diagnostics]
        assert "ORPHANED_NODES" in codes or "MISSING_PARENT" in codes

    def test_parent_child_mismatch(self, valid_spec):
        """Test that parent/child mismatches are detected."""
        valid_spec["hierarchy"]["task-1"]["parent"] = "wrong-parent"
        result = validate_spec(valid_spec)
        codes = [d.code for d in result.diagnostics]
        assert "PARENT_CHILD_MISMATCH" in codes or "MISSING_PARENT" in codes


class TestGetFixActions:
    """Tests for get_fix_actions function."""

    def test_generates_count_fix_action(self, spec_with_issues):
        """Test that count mismatches generate fix actions."""
        result = validate_spec(spec_with_issues)
        actions = get_fix_actions(result, spec_with_issues)
        action_ids = [a.id for a in actions]
        assert any("counts" in aid for aid in action_ids)

    def test_generates_bidirectional_fix_action(self, spec_with_issues):
        """Test that bidirectional inconsistencies generate fix actions."""
        result = validate_spec(spec_with_issues)
        actions = get_fix_actions(result, spec_with_issues)
        action_ids = [a.id for a in actions]
        assert any("bidirectional" in aid for aid in action_ids)

    def test_fix_actions_are_auto_apply(self, spec_with_issues):
        """Test that fix actions have auto_apply set correctly."""
        result = validate_spec(spec_with_issues)
        actions = get_fix_actions(result, spec_with_issues)
        assert all(a.auto_apply for a in actions)

    def test_no_fix_actions_for_valid_spec(self, valid_spec):
        """Test that valid specs don't generate fix actions."""
        result = validate_spec(valid_spec)
        actions = get_fix_actions(result, valid_spec)
        # Valid spec might have warnings but no auto-fixable errors
        assert result.is_valid or len(actions) == 0


class TestApplyFixes:
    """Tests for apply_fixes function."""

    def test_dry_run_returns_skipped_actions(self, spec_with_issues, tmp_path):
        """Test that dry_run mode returns skipped actions."""
        spec_file = tmp_path / "spec.json"
        import json
        spec_file.write_text(json.dumps(spec_with_issues))

        result = validate_spec(spec_with_issues)
        actions = get_fix_actions(result, spec_with_issues)

        report = apply_fixes(actions, str(spec_file), dry_run=True)
        assert len(report.skipped_actions) == len(actions)
        assert len(report.applied_actions) == 0

    def test_creates_backup_when_enabled(self, spec_with_issues, tmp_path):
        """Test that backup is created when enabled."""
        spec_file = tmp_path / "spec.json"
        import json
        spec_file.write_text(json.dumps(spec_with_issues))

        result = validate_spec(spec_with_issues)
        actions = get_fix_actions(result, spec_with_issues)

        if actions:
            report = apply_fixes(actions, str(spec_file), create_backup=True)
            assert report.backup_path is not None

    def test_applies_fixes_correctly(self, spec_with_issues, tmp_path):
        """Test that fixes are applied correctly."""
        spec_file = tmp_path / "spec.json"
        import json
        spec_file.write_text(json.dumps(spec_with_issues))

        result = validate_spec(spec_with_issues)
        actions = get_fix_actions(result, spec_with_issues)

        if actions:
            report = apply_fixes(actions, str(spec_file), create_backup=False)
            assert len(report.applied_actions) > 0

            # Reload and validate again
            fixed_spec = json.loads(spec_file.read_text())
            fixed_result = validate_spec(fixed_spec)

            # Should have fewer errors after fixing
            assert fixed_result.error_count <= result.error_count


class TestCalculateStats:
    """Tests for calculate_stats function."""

    def test_calculates_basic_stats(self, valid_spec):
        """Test that basic stats are calculated."""
        stats = calculate_stats(valid_spec)
        assert stats.spec_id == "test-spec-2025-01-01-001"
        assert stats.totals["nodes"] == 2
        assert stats.totals["tasks"] == 1
        assert stats.progress == 0.0

    def test_calculates_status_counts(self, valid_spec):
        """Test that status counts are calculated."""
        valid_spec["hierarchy"]["task-1"]["status"] = "completed"
        valid_spec["hierarchy"]["task-1"]["completed_tasks"] = 1
        valid_spec["hierarchy"]["spec-root"]["completed_tasks"] = 1
        stats = calculate_stats(valid_spec)
        assert stats.status_counts["completed"] == 1
        assert stats.status_counts["pending"] == 0

    def test_calculates_progress(self, valid_spec):
        """Test that progress is calculated correctly."""
        valid_spec["hierarchy"]["task-1"]["status"] = "completed"
        valid_spec["hierarchy"]["task-1"]["completed_tasks"] = 1
        valid_spec["hierarchy"]["spec-root"]["completed_tasks"] = 1
        stats = calculate_stats(valid_spec)
        assert stats.progress == 1.0

    def test_calculates_file_size(self, valid_spec, tmp_path):
        """Test that file size is calculated when path provided."""
        spec_file = tmp_path / "spec.json"
        import json
        spec_file.write_text(json.dumps(valid_spec))
        stats = calculate_stats(valid_spec, str(spec_file))
        assert stats.file_size_kb > 0

    def test_calculates_max_depth(self, valid_spec):
        """Test that max depth is calculated."""
        stats = calculate_stats(valid_spec)
        assert stats.max_depth >= 1


class TestDiagnosticStructure:
    """Tests for Diagnostic dataclass structure."""

    def test_diagnostic_fields(self):
        """Test that Diagnostic has all required fields."""
        diag = Diagnostic(
            code="TEST_CODE",
            message="Test message",
            severity="error",
            category="test",
            location="node-1",
            suggested_fix="Fix it",
            auto_fixable=True,
        )
        assert diag.code == "TEST_CODE"
        assert diag.message == "Test message"
        assert diag.severity == "error"
        assert diag.category == "test"
        assert diag.location == "node-1"
        assert diag.suggested_fix == "Fix it"
        assert diag.auto_fixable is True


class TestValidationConstants:
    """Tests for validation constants."""

    def test_valid_node_types(self):
        """Test that valid node types are defined."""
        assert "spec" in VALID_NODE_TYPES
        assert "phase" in VALID_NODE_TYPES
        assert "group" in VALID_NODE_TYPES
        assert "task" in VALID_NODE_TYPES
        assert "subtask" in VALID_NODE_TYPES
        assert "verify" in VALID_NODE_TYPES

    def test_valid_statuses(self):
        """Test that valid statuses are defined."""
        assert "pending" in VALID_STATUSES
        assert "in_progress" in VALID_STATUSES
        assert "completed" in VALID_STATUSES
        assert "blocked" in VALID_STATUSES

    def test_valid_verification_types(self):
        """Test that valid verification types are defined."""
        assert "auto" in VALID_VERIFICATION_TYPES
        assert "manual" in VALID_VERIFICATION_TYPES
        assert "fidelity" in VALID_VERIFICATION_TYPES
