"""
Integration tests for spec helper tools.

Tests:
- Response envelope compliance (success/error structure)
- Feature flag integration
- Tool registration and discovery metadata
- End-to-end tool execution with real file system
- Cycle detection scenarios
- Path validation workflows
"""

import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch
import os

import pytest

from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.config import ServerConfig


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            name = func.__name__
            mcp._tools[name] = MagicMock(fn=func)
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def sample_spec_with_cycles():
    """Create a spec with cyclic dependencies."""
    return {
        "spec_id": "spec-with-cycles",
        "metadata": {"title": "Spec with Cycles"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Spec Root",
                "status": "in_progress",
                "children": ["task-a", "task-b"],
            },
            "task-a": {
                "type": "task",
                "title": "Task A",
                "status": "pending",
                "parent": "spec-root",
                "dependencies": {"blocked_by": ["task-b"]},
                "children": [],
            },
            "task-b": {
                "type": "task",
                "title": "Task B",
                "status": "pending",
                "parent": "spec-root",
                "dependencies": {"blocked_by": ["task-a"]},  # Cycle: A -> B -> A
                "children": [],
            },
        },
        "journal": [],
    }


@pytest.fixture
def sample_spec_no_cycles():
    """Create a spec without cyclic dependencies."""
    return {
        "spec_id": "spec-no-cycles",
        "metadata": {"title": "Spec without Cycles"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Spec Root",
                "status": "in_progress",
                "children": ["task-a", "task-b", "task-c"],
            },
            "task-a": {
                "type": "task",
                "title": "Task A",
                "status": "pending",
                "parent": "spec-root",
                "dependencies": {},
                "children": [],
            },
            "task-b": {
                "type": "task",
                "title": "Task B",
                "status": "pending",
                "parent": "spec-root",
                "dependencies": {"blocked_by": ["task-a"]},  # B depends on A
                "children": [],
            },
            "task-c": {
                "type": "task",
                "title": "Task C",
                "status": "pending",
                "parent": "spec-root",
                "dependencies": {"blocked_by": ["task-b"]},  # C depends on B (A -> B -> C)
                "children": [],
            },
        },
        "journal": [],
    }


@pytest.fixture
def sample_spec_with_file_refs():
    """Create a spec with file path references."""
    return {
        "spec_id": "spec-file-refs",
        "metadata": {"title": "Spec with File References"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Spec Root",
                "status": "in_progress",
                "children": ["task-1", "task-2"],
            },
            "task-1": {
                "type": "task",
                "title": "Implement Auth",
                "status": "completed",
                "parent": "spec-root",
                "metadata": {"file_path": "src/services/auth.py"},
                "children": [],
            },
            "task-2": {
                "type": "task",
                "title": "Add User Model",
                "status": "pending",
                "parent": "spec-root",
                "metadata": {"file_path": "src/models/user.py"},
                "children": [],
            },
        },
        "journal": [],
    }


def setup_spec_file(tmp_path, spec_data, spec_id=None):
    """Helper to create spec file in correct location."""
    if spec_id is None:
        spec_id = spec_data.get("spec_id", "test-spec")
    specs_dir = tmp_path / "specs" / "active"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_file = specs_dir / f"{spec_id}.json"
    spec_file.write_text(json.dumps(spec_data))
    return spec_file


class TestSpecHelperResponseEnvelopes:
    """Integration tests for spec helper tool response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test that success responses include required envelope fields."""
        result = asdict(success_response(data={"spec_id": "test-123"}))

        # Required fields per MCP best practices
        assert "success" in result
        assert result["success"] is True
        assert "data" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test that error responses include required envelope fields."""
        result = asdict(
            error_response(
                "Spec not found",
                error_code="SPEC_NOT_FOUND",
                error_type="validation",
            )
        )

        # Required fields per MCP best practices
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "data" in result
        assert result["data"]["error_code"] == "SPEC_NOT_FOUND"
        assert result["data"]["error_type"] == "validation"
        assert "meta" in result

    def test_success_response_with_warnings(self):
        """Test that success responses can include warnings."""
        warnings = ["Some file references may be stale"]
        result = asdict(success_response(data={}, warnings=warnings))

        assert result["success"] is True
        assert "warnings" in result["meta"]
        assert "Some file references may be stale" in result["meta"]["warnings"]

    def test_error_response_with_remediation(self):
        """Test that error responses can include remediation guidance."""
        result = asdict(
            error_response(
                "Circular dependency detected",
                error_code="CYCLE_DETECTED",
                remediation="Review task dependencies and break the cycle.",
            )
        )

        assert result["success"] is False
        assert "remediation" in result["data"]


class TestSpecHelperFeatureFlagIntegration:
    """Integration tests for feature flag system with spec helper tools."""

    def test_spec_helpers_flag_in_manifest(self):
        """Test spec_helpers flag is defined in capabilities manifest."""
        manifest_path = Path(__file__).parent.parent.parent / "mcp" / "capabilities_manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        flags = manifest.get("feature_flags", {}).get("flags", {})

        assert "spec_helpers" in flags
        assert flags["spec_helpers"]["state"] == "beta"
        assert flags["spec_helpers"]["default_enabled"] is True
        assert flags["spec_helpers"]["percentage_rollout"] == 100

    def test_spec_helper_tools_in_manifest(self):
        """Test spec helper tools are registered in capabilities manifest."""
        manifest_path = Path(__file__).parent.parent.parent / "mcp" / "capabilities_manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        tools = manifest.get("tools", {}).get("spec_helper_tools", [])
        tool_names = [t["name"] for t in tools]

        expected_tools = [
            "spec_find_related_files",
            "spec_find_patterns",
            "spec_detect_cycles",
            "spec_validate_paths",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    def test_spec_helper_tools_have_feature_flag_reference(self):
        """Test each spec helper tool references the feature flag."""
        manifest_path = Path(__file__).parent.parent.parent / "mcp" / "capabilities_manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        tools = manifest.get("tools", {}).get("spec_helper_tools", [])

        for tool in tools:
            assert tool.get("feature_flag") == "spec_helpers", (
                f"Tool {tool['name']} missing feature_flag reference"
            )

    def test_server_capabilities_include_spec_helpers(self):
        """Test server capabilities expose spec helper feature."""
        manifest_path = Path(__file__).parent.parent.parent / "mcp" / "capabilities_manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        caps = manifest.get("server_capabilities", {}).get("features", {})

        assert "spec_helpers" in caps
        assert caps["spec_helpers"]["supported"] is True
        assert "tools" in caps["spec_helpers"]


class TestCycleDetectionScenarios:
    """Integration tests for cycle detection scenarios."""

    def test_detect_simple_cycle(self, mock_mcp, tmp_path, sample_spec_with_cycles):
        """Test detection of a simple A->B->A cycle."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        setup_spec_file(tmp_path, sample_spec_with_cycles, "spec-with-cycles")
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        detect_cycles = mock_mcp._tools["spec_detect_cycles"]

        # Change to tmp_path so the tool finds the specs
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = detect_cycles.fn(spec_id="spec-with-cycles")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["has_cycles"] is True
        assert result["data"]["cycle_count"] >= 1
        assert len(result["data"]["affected_tasks"]) >= 2

    def test_no_cycles_in_acyclic_graph(self, mock_mcp, tmp_path, sample_spec_no_cycles):
        """Test no cycles detected in properly structured spec."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        setup_spec_file(tmp_path, sample_spec_no_cycles, "spec-no-cycles")
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        detect_cycles = mock_mcp._tools["spec_detect_cycles"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = detect_cycles.fn(spec_id="spec-no-cycles")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["has_cycles"] is False
        assert result["data"]["cycles"] == []
        assert result["data"]["affected_tasks"] == []

    def test_detect_complex_cycle(self, mock_mcp, tmp_path):
        """Test detection of complex cycle A->B->C->D->B."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        spec_data = {
            "spec_id": "complex-cycle-spec",
            "metadata": {"title": "Complex Cycle Spec"},
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Root",
                    "status": "in_progress",
                    "children": ["task-a", "task-b", "task-c", "task-d"],
                },
                "task-a": {
                    "type": "task",
                    "title": "Task A",
                    "status": "pending",
                    "parent": "spec-root",
                    "dependencies": {},
                    "children": [],
                },
                "task-b": {
                    "type": "task",
                    "title": "Task B",
                    "status": "pending",
                    "parent": "spec-root",
                    "dependencies": {"blocked_by": ["task-a", "task-d"]},  # Cycle: B->D->C->B
                    "children": [],
                },
                "task-c": {
                    "type": "task",
                    "title": "Task C",
                    "status": "pending",
                    "parent": "spec-root",
                    "dependencies": {"blocked_by": ["task-b"]},
                    "children": [],
                },
                "task-d": {
                    "type": "task",
                    "title": "Task D",
                    "status": "pending",
                    "parent": "spec-root",
                    "dependencies": {"blocked_by": ["task-c"]},
                    "children": [],
                },
            },
            "journal": [],
        }

        setup_spec_file(tmp_path, spec_data, "complex-cycle-spec")
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        detect_cycles = mock_mcp._tools["spec_detect_cycles"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = detect_cycles.fn(spec_id="complex-cycle-spec")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["has_cycles"] is True
        assert result["data"]["cycle_count"] >= 1

    def test_detect_cycles_spec_not_found(self, mock_mcp, tmp_path):
        """Test cycle detection with non-existent spec."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create specs dir but no spec file
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        detect_cycles = mock_mcp._tools["spec_detect_cycles"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = detect_cycles.fn(spec_id="nonexistent-spec")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is False
        assert "SPEC_NOT_FOUND" in result.get("data", {}).get("error_code", "") or \
               "not found" in result.get("error", "").lower()


class TestPathValidationWorkflows:
    """Integration tests for path validation workflows."""

    def test_validate_existing_paths(self, mock_mcp, tmp_path):
        """Test validation of paths that exist."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create test files
        src_path = tmp_path / "src"
        src_path.mkdir()
        (src_path / "main.py").touch()
        (src_path / "utils.py").touch()

        # Also create specs dir for the tool to find
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        validate_paths = mock_mcp._tools["spec_validate_paths"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = validate_paths.fn(
                paths=["src/main.py", "src/utils.py"],
                base_directory=str(tmp_path)
            )
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["all_valid"] is True
        assert result["data"]["valid_count"] == 2
        assert result["data"]["invalid_count"] == 0

    def test_validate_mixed_paths(self, mock_mcp, tmp_path):
        """Test validation with mix of valid and invalid paths."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create only one file
        src_path = tmp_path / "src"
        src_path.mkdir()
        (src_path / "existing.py").touch()

        # Also create specs dir
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        validate_paths = mock_mcp._tools["spec_validate_paths"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = validate_paths.fn(
                paths=["src/existing.py", "src/missing.py", "src/deleted.py"],
                base_directory=str(tmp_path)
            )
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["all_valid"] is False
        assert result["data"]["valid_count"] == 1
        assert result["data"]["invalid_count"] == 2

    def test_validate_empty_path_list_returns_error(self, mock_mcp, tmp_path):
        """Test validation with empty path list returns error."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create specs dir
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        validate_paths = mock_mcp._tools["spec_validate_paths"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = validate_paths.fn(paths=[])
        finally:
            os.chdir(old_cwd)

        # Empty paths list now returns error per implementation
        assert result["success"] is False
        assert "MISSING_REQUIRED" in result.get("data", {}).get("error_code", "")


class TestPatternSearchWorkflows:
    """Integration tests for pattern search workflows."""

    def test_find_python_files(self, mock_mcp, tmp_path):
        """Test finding Python files with glob pattern."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create test files
        src_path = tmp_path / "src"
        src_path.mkdir()
        (src_path / "main.py").touch()
        (src_path / "utils.py").touch()
        (src_path / "helpers.py").touch()
        (src_path / "readme.md").touch()  # Non-Python file

        # Create specs dir
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        find_patterns = mock_mcp._tools["spec_find_patterns"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = find_patterns.fn(pattern="src/*.py")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["total_count"] == 3
        assert "main.py" in str(result["data"]["matches"])
        assert "utils.py" in str(result["data"]["matches"])
        assert "helpers.py" in str(result["data"]["matches"])

    def test_find_files_in_subdirectory(self, mock_mcp, tmp_path):
        """Test finding files in subdirectory."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create nested test files
        tests_path = tmp_path / "tests" / "unit"
        tests_path.mkdir(parents=True)
        (tests_path / "test_auth.py").touch()
        (tests_path / "test_utils.py").touch()

        # Create specs dir
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        find_patterns = mock_mcp._tools["spec_find_patterns"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = find_patterns.fn(pattern="tests/**/*.py", directory=str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["total_count"] == 2

    def test_find_no_matches(self, mock_mcp, tmp_path):
        """Test pattern that matches no files."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create specs dir but no matching files
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        find_patterns = mock_mcp._tools["spec_find_patterns"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = find_patterns.fn(pattern="*.nonexistent")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert result["data"]["total_count"] == 0
        assert result["data"]["matches"] == []


class TestRelatedFilesWorkflows:
    """Integration tests for related files discovery workflows."""

    def test_find_files_related_to_source(self, mock_mcp, tmp_path, sample_spec_with_file_refs):
        """Test finding files related to a source file."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        setup_spec_file(tmp_path, sample_spec_with_file_refs, "spec-file-refs")
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        find_related = mock_mcp._tools["spec_find_related_files"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = find_related.fn(file_path="src/services/auth.py")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        # Should find related files from the spec
        assert "related_files" in result["data"]
        assert "total_count" in result["data"]

    def test_find_related_files_scoped_to_spec(
        self, mock_mcp, tmp_path, sample_spec_with_file_refs
    ):
        """Test finding related files scoped to a specific spec."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        setup_spec_file(tmp_path, sample_spec_with_file_refs, "spec-file-refs")
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        find_related = mock_mcp._tools["spec_find_related_files"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = find_related.fn(
                file_path="src/services/auth.py",
                spec_id="spec-file-refs"
            )
        finally:
            os.chdir(old_cwd)

        assert result["success"] is True
        assert "related_files" in result["data"]

    def test_find_related_files_missing_file_path(self, mock_mcp, tmp_path):
        """Test error when file_path is not provided."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create specs dir
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_spec_helper_tools(mock_mcp, config)

        find_related = mock_mcp._tools["spec_find_related_files"]

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = find_related.fn(file_path="")
        finally:
            os.chdir(old_cwd)

        assert result["success"] is False
        assert "MISSING_REQUIRED" in result.get("data", {}).get("error_code", "")


class TestEndToEndWorkflow:
    """Integration tests for end-to-end spec helper workflows."""

    def test_spec_validation_workflow(self):
        """Test a typical spec validation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal spec structure
            spec_path = Path(tmpdir) / "specs" / "active"
            spec_path.mkdir(parents=True)

            spec_file = spec_path / "test-spec-001.json"
            spec_file.write_text(json.dumps({
                "spec_id": "test-spec-001",
                "title": "Test Specification",
                "phases": [
                    {
                        "id": "phase-1",
                        "tasks": [
                            {"id": "task-1-1", "title": "First task"},
                            {"id": "task-1-2", "title": "Second task"},
                        ]
                    }
                ]
            }))

            # Verify file exists
            assert spec_file.exists()
            content = json.loads(spec_file.read_text())
            assert content["spec_id"] == "test-spec-001"

    def test_file_reference_audit_workflow(self):
        """Test auditing file references in a spec."""
        # Simulate checking if all referenced files exist
        spec_file_paths = [
            "src/services/auth.py",
            "src/models/user.py",
            "tests/test_auth.py",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only some of the files
            src_path = Path(tmpdir) / "src"
            src_path.mkdir()
            (src_path / "services").mkdir()
            (src_path / "services" / "auth.py").touch()
            (src_path / "models").mkdir()
            (src_path / "models" / "user.py").touch()
            # Note: tests/test_auth.py not created

            # Simulate validation result
            existing = []
            missing = []
            for path in spec_file_paths:
                full_path = Path(tmpdir) / path
                if full_path.exists():
                    existing.append(path)
                else:
                    missing.append(path)

            assert len(existing) == 2
            assert len(missing) == 1
            assert "tests/test_auth.py" in missing


class TestToolRegistration:
    """Test all spec helper tools register correctly."""

    def test_all_spec_helper_tools_register(self, mock_mcp):
        """Test all spec helper tools register without error."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        config = ServerConfig(specs_dir=Path("/tmp/specs"))

        # Should not raise
        register_spec_helper_tools(mock_mcp, config)

        # Should have registered expected tools
        assert "spec_find_related_files" in mock_mcp._tools
        assert "spec_find_patterns" in mock_mcp._tools
        assert "spec_detect_cycles" in mock_mcp._tools
        assert "spec_validate_paths" in mock_mcp._tools
