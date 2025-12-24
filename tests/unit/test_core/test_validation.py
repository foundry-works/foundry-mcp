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
    add_verification,
    execute_verification,
    format_verification_summary,
    Diagnostic,
    VALID_NODE_TYPES,
    VALID_STATUSES,
    VALID_VERIFICATION_TYPES,
    VERIFICATION_RESULTS,
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
def medium_spec(valid_spec):
    """Return a medium-complexity spec with required task fields."""
    spec = copy.deepcopy(valid_spec)
    spec["metadata"] = {"template": "medium", "mission": "Ship the feature"}
    task_metadata = spec["hierarchy"]["task-1"]["metadata"]
    task_metadata["description"] = "Implement the core task"
    task_metadata["acceptance_criteria"] = ["Core behavior matches requirements"]
    return spec


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
                "metadata": {
                    "task_category": "implementation",
                    "file_path": "test2.py",
                },
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },  # Missing blocked_by
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

    def test_missing_mission_for_medium_spec(self, medium_spec):
        """Test that medium specs require a mission."""
        medium_spec["metadata"]["mission"] = ""
        result = validate_spec(medium_spec)
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_MISSION" in codes

    def test_missing_task_category_for_medium_spec(self, medium_spec):
        """Test that medium specs require task_category on tasks."""
        del medium_spec["hierarchy"]["task-1"]["metadata"]["task_category"]
        result = validate_spec(medium_spec)
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_TASK_CATEGORY" in codes

    def test_missing_task_description_for_medium_spec(self, medium_spec):
        """Test that medium specs require descriptions on tasks."""
        del medium_spec["hierarchy"]["task-1"]["metadata"]["description"]
        result = validate_spec(medium_spec)
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_TASK_DESCRIPTION" in codes

    def test_missing_acceptance_criteria_for_medium_spec(self, medium_spec):
        """Test that medium specs require acceptance criteria on tasks."""
        del medium_spec["hierarchy"]["task-1"]["metadata"]["acceptance_criteria"]
        result = validate_spec(medium_spec)
        codes = [d.code for d in result.diagnostics]
        assert "MISSING_ACCEPTANCE_CRITERIA" in codes

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

    def test_missing_file_path_is_not_auto_fixable(self, valid_spec):
        """Missing file_path should not be auto-fixed with placeholders."""
        del valid_spec["hierarchy"]["task-1"]["metadata"]["file_path"]

        result = validate_spec(valid_spec)
        diags = [d for d in result.diagnostics if d.code == "MISSING_FILE_PATH"]
        assert diags, "Expected MISSING_FILE_PATH diagnostic"
        assert all(not d.auto_fixable for d in diags)

        actions = get_fix_actions(result, valid_spec)
        assert not any(a.id.startswith("metadata.add_file_path:") for a in actions)

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
        # Canonical values are run-tests, fidelity, and manual
        assert "run-tests" in VALID_VERIFICATION_TYPES
        assert "fidelity" in VALID_VERIFICATION_TYPES
        assert "manual" in VALID_VERIFICATION_TYPES
        # Legacy values should NOT be present
        assert "test" not in VALID_VERIFICATION_TYPES
        assert "auto" not in VALID_VERIFICATION_TYPES


class TestAddVerification:
    """Tests for add_verification function."""

    @pytest.fixture
    def spec_with_verify_node(self):
        """Return a spec with a verify node for testing."""
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
                    "children": ["verify-1"],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {},
                },
                "verify-1": {
                    "type": "verify",
                    "title": "Test Verification",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": [],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {"verification_type": "run-tests", "command": "echo test"},
                },
            },
        }

    def test_add_verification_success(self, spec_with_verify_node):
        """Test successful verification result addition."""
        success, error = add_verification(
            spec_data=spec_with_verify_node,
            verify_id="verify-1",
            result="PASSED",
            command="echo test",
            output="test output",
        )
        assert success is True
        assert error is None

        # Check metadata was updated
        metadata = spec_with_verify_node["hierarchy"]["verify-1"]["metadata"]
        assert metadata["last_result"] == "PASSED"
        assert "last_verified_at" in metadata
        assert len(metadata["verification_history"]) == 1

    def test_add_verification_invalid_result(self, spec_with_verify_node):
        """Test that invalid result values are rejected."""
        success, error = add_verification(
            spec_data=spec_with_verify_node,
            verify_id="verify-1",
            result="INVALID",
        )
        assert success is False
        assert error is not None
        assert "Invalid result" in error

    def test_add_verification_node_not_found(self, spec_with_verify_node):
        """Test error when verify node doesn't exist."""
        success, error = add_verification(
            spec_data=spec_with_verify_node,
            verify_id="nonexistent",
            result="PASSED",
        )
        assert success is False
        assert error is not None
        assert "not found" in error

    def test_add_verification_wrong_node_type(self, spec_with_verify_node):
        """Test error when node is not a verify type."""
        spec_with_verify_node["hierarchy"]["verify-1"]["type"] = "task"
        success, error = add_verification(
            spec_data=spec_with_verify_node,
            verify_id="verify-1",
            result="PASSED",
        )
        assert success is False
        assert error is not None
        assert "expected 'verify'" in error

    def test_add_verification_history_limit(self, spec_with_verify_node):
        """Test that verification history is limited to 10 entries."""
        for i in range(15):
            add_verification(
                spec_data=spec_with_verify_node,
                verify_id="verify-1",
                result="PASSED",
            )

        metadata = spec_with_verify_node["hierarchy"]["verify-1"]["metadata"]
        assert len(metadata["verification_history"]) == 10

    def test_add_verification_with_all_fields(self, spec_with_verify_node):
        """Test verification with all optional fields."""
        success, error = add_verification(
            spec_data=spec_with_verify_node,
            verify_id="verify-1",
            result="PARTIAL",
            command="pytest tests/",
            output="5 passed, 2 failed",
            issues="Test failures in auth module",
            notes="Needs investigation",
        )
        assert success is True

        history = spec_with_verify_node["hierarchy"]["verify-1"]["metadata"][
            "verification_history"
        ]
        entry = history[0]
        assert entry["result"] == "PARTIAL"
        assert entry["command"] == "pytest tests/"
        assert entry["output"] == "5 passed, 2 failed"
        assert entry["issues"] == "Test failures in auth module"
        assert entry["notes"] == "Needs investigation"


class TestExecuteVerification:
    """Tests for execute_verification function."""

    @pytest.fixture
    def spec_with_echo_command(self):
        """Return a spec with a verify node that echoes a message."""
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
                    "children": ["verify-1"],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {},
                },
                "verify-1": {
                    "type": "verify",
                    "title": "Echo Verification",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": [],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {
                        "verification_type": "run-tests",
                        "command": "echo hello world",
                    },
                },
            },
        }

    @pytest.fixture
    def spec_with_failing_command(self):
        """Return a spec with a verify node that has a failing command."""
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
                    "children": ["verify-1"],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {},
                },
                "verify-1": {
                    "type": "verify",
                    "title": "Failing Verification",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": [],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {"verification_type": "run-tests", "command": "exit 1"},
                },
            },
        }

    def test_execute_verification_success(self, spec_with_echo_command):
        """Test successful command execution."""
        result = execute_verification(spec_with_echo_command, "verify-1")
        assert result["success"] is True
        assert result["result"] == "PASSED"
        assert result["exit_code"] == 0
        assert "hello world" in result["output"]
        assert result["command"] == "echo hello world"

    def test_execute_verification_failing_command(self, spec_with_failing_command):
        """Test execution of a failing command."""
        result = execute_verification(spec_with_failing_command, "verify-1")
        assert (
            result["success"] is True
        )  # Execution completed, even if result is FAILED
        assert result["result"] == "FAILED"
        assert result["exit_code"] == 1

    def test_execute_verification_node_not_found(self, spec_with_echo_command):
        """Test error when verify node doesn't exist."""
        result = execute_verification(spec_with_echo_command, "nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_execute_verification_wrong_node_type(self, spec_with_echo_command):
        """Test error when node is not a verify type."""
        spec_with_echo_command["hierarchy"]["verify-1"]["type"] = "task"
        result = execute_verification(spec_with_echo_command, "verify-1")
        assert result["success"] is False
        assert "expected 'verify'" in result["error"]

    def test_execute_verification_no_command(self, spec_with_echo_command):
        """Test error when no command is defined."""
        del spec_with_echo_command["hierarchy"]["verify-1"]["metadata"]["command"]
        result = execute_verification(spec_with_echo_command, "verify-1")
        assert result["success"] is False
        assert "No command defined" in result["error"]

    def test_execute_verification_with_record(self, spec_with_echo_command):
        """Test that results are recorded when record=True."""
        result = execute_verification(spec_with_echo_command, "verify-1", record=True)
        assert result["success"] is True
        assert result["recorded"] is True

        # Check the verification was recorded
        metadata = spec_with_echo_command["hierarchy"]["verify-1"]["metadata"]
        assert metadata["last_result"] == "PASSED"
        assert len(metadata["verification_history"]) == 1

    def test_execute_verification_without_record(self, spec_with_echo_command):
        """Test that results are not recorded when record=False."""
        result = execute_verification(spec_with_echo_command, "verify-1", record=False)
        assert result["success"] is True
        assert result["recorded"] is False

        # Check no verification was recorded
        metadata = spec_with_echo_command["hierarchy"]["verify-1"]["metadata"]
        assert "last_result" not in metadata
        assert "verification_history" not in metadata

    def test_execute_verification_timeout(self, spec_with_echo_command):
        """Test command timeout handling."""
        # Change command to sleep longer than timeout
        spec_with_echo_command["hierarchy"]["verify-1"]["metadata"]["command"] = (
            "sleep 5"
        )
        result = execute_verification(spec_with_echo_command, "verify-1", timeout=1)
        assert "timed out" in result["error"]
        assert result["result"] == "FAILED"
        assert result["exit_code"] == -1

    def test_execute_verification_invalid_hierarchy(self):
        """Test error with invalid hierarchy."""
        result = execute_verification({"spec_id": "test"}, "verify-1")
        assert result["success"] is False
        assert "missing or invalid hierarchy" in result["error"]

    def test_execute_verification_returns_spec_id(self, spec_with_echo_command):
        """Test that result includes spec_id."""
        result = execute_verification(spec_with_echo_command, "verify-1")
        assert result["spec_id"] == "test-spec-2025-01-01-001"

    def test_execute_verification_captures_stderr(self, spec_with_echo_command):
        """Test that stderr is captured."""
        spec_with_echo_command["hierarchy"]["verify-1"]["metadata"]["command"] = (
            "echo error >&2"
        )
        result = execute_verification(spec_with_echo_command, "verify-1")
        assert result["success"] is True
        assert "[stderr]" in result["output"]
        assert "error" in result["output"]


class TestFormatVerificationSummary:
    """Tests for format_verification_summary function."""

    def test_format_single_passed_verification(self):
        """Test formatting a single passed verification result."""
        result = {
            "verify_id": "verify-1",
            "result": "PASSED",
            "command": "echo test",
            "output": "test output",
        }
        summary = format_verification_summary(result)
        assert summary["total_verifications"] == 1
        assert summary["passed"] == 1
        assert summary["failed"] == 0
        assert summary["partial"] == 0
        assert "Verification Summary: 1 total" in summary["summary"]
        assert "✓ Passed:  1" in summary["summary"]

    def test_format_multiple_verifications(self):
        """Test formatting multiple verification results."""
        results = [
            {"verify_id": "verify-1", "result": "PASSED", "command": "test1"},
            {"verify_id": "verify-2", "result": "FAILED", "command": "test2"},
            {"verify_id": "verify-3", "result": "PARTIAL", "command": "test3"},
        ]
        summary = format_verification_summary(results)
        assert summary["total_verifications"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["partial"] == 1
        assert "Verification Summary: 3 total" in summary["summary"]

    def test_format_verifications_dict_input(self):
        """Test formatting from dict with verifications key."""
        data = {
            "verifications": [
                {"verify_id": "verify-1", "result": "PASSED"},
                {"verify_id": "verify-2", "result": "PASSED"},
            ]
        }
        summary = format_verification_summary(data)
        assert summary["total_verifications"] == 2
        assert summary["passed"] == 2

    def test_format_empty_list(self):
        """Test formatting empty verification list."""
        summary = format_verification_summary([])
        assert summary["total_verifications"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert "Verification Summary: 0 total" in summary["summary"]

    def test_format_includes_error_messages(self):
        """Test that error messages are included in summary."""
        result = {
            "verify_id": "verify-1",
            "result": "FAILED",
            "error": "Command timed out",
        }
        summary = format_verification_summary(result)
        assert "Error: Command timed out" in summary["summary"]

    def test_format_truncates_long_output(self):
        """Test that long output is truncated in preview."""
        result = {
            "verify_id": "verify-1",
            "result": "PASSED",
            "output": "x" * 500,  # Long output
        }
        summary = format_verification_summary(result)
        result_entry = summary["results"][0]
        assert len(result_entry["output_preview"]) < 210  # 200 + "..."

    def test_format_truncates_long_command(self):
        """Test that long commands are truncated in summary text."""
        result = {
            "verify_id": "verify-1",
            "result": "PASSED",
            "command": "very_long_command_" * 10,  # Long command
        }
        summary = format_verification_summary(result)
        assert "..." in summary["summary"]  # Command truncated

    def test_format_handles_unknown_result(self):
        """Test handling of unknown result types."""
        result = {
            "verify_id": "verify-1",
            "result": None,
        }
        summary = format_verification_summary(result)
        assert summary["total_verifications"] == 1
        assert summary["results"][0]["result"] == "UNKNOWN"
        assert summary["results"][0]["status_icon"] == "?"

    def test_format_results_contain_all_fields(self):
        """Test that result entries contain all expected fields."""
        result = {
            "verify_id": "verify-1",
            "result": "PASSED",
            "command": "echo test",
            "output": "test output",
        }
        summary = format_verification_summary(result)
        entry = summary["results"][0]
        assert "verify_id" in entry
        assert "result" in entry
        assert "status_icon" in entry
        assert "command" in entry
        assert "output_preview" in entry

    def test_format_skips_non_dict_items(self):
        """Test that non-dict items in list are skipped."""
        results = [
            {"verify_id": "verify-1", "result": "PASSED"},
            "invalid",
            None,
            {"verify_id": "verify-2", "result": "FAILED"},
        ]
        summary = format_verification_summary(results)
        assert summary["total_verifications"] == 2


class TestVerificationConstants:
    """Tests for verification constants."""

    def test_verification_results_constant(self):
        """Test that verification results constant is defined."""
        assert "PASSED" in VERIFICATION_RESULTS
        assert "FAILED" in VERIFICATION_RESULTS
        assert "PARTIAL" in VERIFICATION_RESULTS
