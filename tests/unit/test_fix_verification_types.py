"""
Unit tests for task fix-verification-types handler.

Tests verification type auto-fixer with dry-run, mapping, and persistence.
"""

import json
import pytest
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig
from tests.conftest import extract_response_dict


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_specs_dir(tmp_path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    for d in ["active", "pending", "completed", "archived"]:
        (specs_dir / d).mkdir()
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    return ServerConfig(server_name="test", server_version="0.1.0", specs_dir=test_specs_dir, log_level="WARNING")


@pytest.fixture
def task_tool(test_config):
    raw_fn = create_server(test_config)._tool_manager._tools["task"].fn
    def wrapper(*args, **kwargs):
        return extract_response_dict(raw_fn(*args, **kwargs))
    return wrapper


def write_spec(specs_dir, spec_id: str, hierarchy: dict, status: str = "active"):
    """Helper to write a spec file with given hierarchy."""
    spec = {
        "spec_id": spec_id,
        "title": "Test Spec",
        "metadata": {"title": "Test Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": hierarchy,
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (specs_dir / status / f"{spec_id}.json").write_text(json.dumps(spec))
    return spec


def read_spec(specs_dir, spec_id: str, status: str = "active") -> dict:
    """Helper to read a spec file."""
    return json.loads((specs_dir / status / f"{spec_id}.json").read_text())


# =============================================================================
# Required Parameter Tests
# =============================================================================

class TestRequiredParams:
    def test_missing_spec_id(self, task_tool):
        result = task_tool(action="fix-verification-types")
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_empty_spec_id(self, task_tool):
        result = task_tool(action="fix-verification-types", spec_id="")
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_whitespace_spec_id(self, task_tool):
        result = task_tool(action="fix-verification-types", spec_id="   ")
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()


# =============================================================================
# Dry Run Tests
# =============================================================================

class TestDryRun:
    def test_dry_run_does_not_persist(self, task_tool, test_specs_dir):
        """Dry run should report fixes but not save changes."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Test verify", "status": "pending", "parent": "spec-root", "children": [], "metadata": {}},
        }
        write_spec(test_specs_dir, "dry-run-test-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="dry-run-test-001", dry_run=True)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["total_fixes"] == 1
        assert result["data"]["applied_count"] == 0

        # Verify file was NOT modified
        spec = read_spec(test_specs_dir, "dry-run-test-001")
        assert spec["hierarchy"]["verify-1"]["metadata"].get("verification_type") is None

    def test_dry_run_false_persists(self, task_tool, test_specs_dir):
        """Non-dry-run should persist changes."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Test verify", "status": "pending", "parent": "spec-root", "children": [], "metadata": {}},
        }
        write_spec(test_specs_dir, "persist-test-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="persist-test-001", dry_run=False)
        assert result["success"] is True
        assert result["data"]["dry_run"] is False
        assert result["data"]["total_fixes"] == 1
        assert result["data"]["applied_count"] == 1

        # Verify file WAS modified
        spec = read_spec(test_specs_dir, "persist-test-001")
        assert spec["hierarchy"]["verify-1"]["metadata"]["verification_type"] == "run-tests"


# =============================================================================
# Missing Verification Type Tests
# =============================================================================

class TestMissingVerificationType:
    def test_missing_type_defaults_to_run_tests(self, task_tool, test_specs_dir):
        """Missing verification_type should default to run-tests."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Run tests", "status": "pending", "parent": "spec-root", "children": [], "metadata": {}},
        }
        write_spec(test_specs_dir, "missing-type-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="missing-type-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 1
        assert result["data"]["fixes"][0]["issue"] == "missing"
        assert result["data"]["fixes"][0]["new_value"] == "run-tests"

    def test_missing_metadata_dict_created(self, task_tool, test_specs_dir):
        """Should create metadata dict if missing entirely."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Run tests", "status": "pending", "parent": "spec-root", "children": []},
        }
        write_spec(test_specs_dir, "no-metadata-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="no-metadata-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 1

        spec = read_spec(test_specs_dir, "no-metadata-001")
        assert "metadata" in spec["hierarchy"]["verify-1"]
        assert spec["hierarchy"]["verify-1"]["metadata"]["verification_type"] == "run-tests"


# =============================================================================
# Legacy Type Mapping Tests
# =============================================================================

class TestLegacyTypeMapping:
    def test_legacy_test_maps_to_run_tests(self, task_tool, test_specs_dir):
        """Legacy 'test' type should map to 'run-tests'."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Run tests", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "test"}},
        }
        write_spec(test_specs_dir, "legacy-test-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="legacy-test-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 1
        assert result["data"]["fixes"][0]["issue"] == "legacy"
        assert result["data"]["fixes"][0]["old_value"] == "test"
        assert result["data"]["fixes"][0]["new_value"] == "run-tests"

    def test_legacy_auto_maps_to_run_tests(self, task_tool, test_specs_dir):
        """Legacy 'auto' type should map to 'run-tests'."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Auto verify", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "auto"}},
        }
        write_spec(test_specs_dir, "legacy-auto-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="legacy-auto-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 1
        assert result["data"]["fixes"][0]["issue"] == "legacy"
        assert result["data"]["fixes"][0]["old_value"] == "auto"
        assert result["data"]["fixes"][0]["new_value"] == "run-tests"

    def test_legacy_mappings_returned_in_response(self, task_tool, test_specs_dir):
        """Response should include available legacy mappings."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Test", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "run-tests"}},
        }
        write_spec(test_specs_dir, "mappings-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="mappings-001")
        assert result["success"] is True
        assert "legacy_mappings" in result["data"]
        assert result["data"]["legacy_mappings"]["test"] == "run-tests"
        assert result["data"]["legacy_mappings"]["auto"] == "run-tests"


# =============================================================================
# Invalid Type Tests
# =============================================================================

class TestInvalidType:
    def test_unknown_type_defaults_to_manual(self, task_tool, test_specs_dir):
        """Unknown verification_type should default to manual."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Custom verify", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "unknown-type"}},
        }
        write_spec(test_specs_dir, "invalid-type-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="invalid-type-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 1
        assert result["data"]["fixes"][0]["issue"] == "invalid"
        assert result["data"]["fixes"][0]["old_value"] == "unknown-type"
        assert result["data"]["fixes"][0]["new_value"] == "manual"

    def test_empty_string_type_treated_as_invalid(self, task_tool, test_specs_dir):
        """Empty string verification_type should be fixed."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Empty verify", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": ""}},
        }
        write_spec(test_specs_dir, "empty-type-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="empty-type-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 1


# =============================================================================
# Valid Types Tests
# =============================================================================

class TestValidTypes:
    def test_run_tests_not_changed(self, task_tool, test_specs_dir):
        """Valid 'run-tests' type should not be changed."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Run tests", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "run-tests"}},
        }
        write_spec(test_specs_dir, "valid-run-tests-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="valid-run-tests-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 0

    def test_fidelity_not_changed(self, task_tool, test_specs_dir):
        """Valid 'fidelity' type should not be changed."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Fidelity check", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "fidelity"}},
        }
        write_spec(test_specs_dir, "valid-fidelity-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="valid-fidelity-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 0

    def test_manual_not_changed(self, task_tool, test_specs_dir):
        """Valid 'manual' type should not be changed."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Manual check", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "manual"}},
        }
        write_spec(test_specs_dir, "valid-manual-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="valid-manual-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 0

    def test_valid_types_returned_in_response(self, task_tool, test_specs_dir):
        """Response should include list of valid types."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Test", "status": "pending", "parent": "spec-root", "children": [], "metadata": {"verification_type": "run-tests"}},
        }
        write_spec(test_specs_dir, "valid-types-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="valid-types-001")
        assert result["success"] is True
        assert "valid_types" in result["data"]
        assert set(result["data"]["valid_types"]) == {"run-tests", "fidelity", "manual"}


# =============================================================================
# Multiple Nodes Tests
# =============================================================================

class TestMultipleNodes:
    def test_fixes_multiple_verify_nodes(self, task_tool, test_specs_dir):
        """Should fix all verify nodes in hierarchy."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["phase-1"]},
            "phase-1": {"type": "phase", "title": "Phase 1", "status": "pending", "parent": "spec-root", "children": ["task-1", "verify-1", "verify-2", "verify-3"]},
            "task-1": {"type": "task", "title": "Task", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "verify-1": {"type": "verify", "title": "V1", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "verify-2": {"type": "verify", "title": "V2", "status": "pending", "parent": "phase-1", "children": [], "metadata": {"verification_type": "test"}},
            "verify-3": {"type": "verify", "title": "V3", "status": "pending", "parent": "phase-1", "children": [], "metadata": {"verification_type": "unknown"}},
        }
        write_spec(test_specs_dir, "multi-node-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="multi-node-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 3
        assert result["data"]["summary"]["missing_set_to_run_tests"] == 1
        assert result["data"]["summary"]["legacy_mapped"] == 1
        assert result["data"]["summary"]["invalid_set_to_manual"] == 1

    def test_skips_non_verify_nodes(self, task_tool, test_specs_dir):
        """Should only process verify nodes, not task/phase nodes."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["phase-1"]},
            "phase-1": {"type": "phase", "title": "Phase 1", "status": "pending", "parent": "spec-root", "children": ["task-1"]},
            "task-1": {"type": "task", "title": "Task", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
        }
        write_spec(test_specs_dir, "no-verify-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="no-verify-001")
        assert result["success"] is True
        assert result["data"]["total_fixes"] == 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    def test_spec_not_found(self, task_tool, test_specs_dir):
        """Should return error for non-existent spec."""
        result = task_tool(action="fix-verification-types", spec_id="nonexistent-spec-999")
        assert result["success"] is False

    def test_invalid_dry_run_type(self, task_tool, test_specs_dir):
        """Should validate dry_run is boolean."""
        hierarchy = {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["verify-1"]},
            "verify-1": {"type": "verify", "title": "Test", "status": "pending", "parent": "spec-root", "children": [], "metadata": {}},
        }
        write_spec(test_specs_dir, "invalid-dry-run-001", hierarchy)

        result = task_tool(action="fix-verification-types", spec_id="invalid-dry-run-001", dry_run="yes")
        assert result["success"] is False
        assert "dry_run" in result["error"].lower()
