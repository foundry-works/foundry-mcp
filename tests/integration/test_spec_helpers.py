"""
Integration tests for spec helper tools.

Tests:
- Response envelope compliance (success/error structure)
- Feature flag integration
- Tool registration and discovery metadata
- End-to-end tool execution with mocked CLI
- Cycle detection scenarios
- Path validation workflows
"""

import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.responses import success_response, error_response


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

    def test_detect_simple_cycle(self):
        """Test detection of a simple A->B->A cycle."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [["task-a", "task-b", "task-a"]],
            "affected_tasks": ["task-a", "task-b"],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-detect-cycles"]("spec-with-cycle")

            assert result["success"] is True
            assert result["data"]["has_cycles"] is True
            assert len(result["data"]["cycles"]) == 1
            assert set(result["data"]["affected_tasks"]) == {"task-a", "task-b"}

    def test_detect_complex_cycle(self):
        """Test detection of complex cycle A->B->C->D->B."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [
                ["task-b", "task-c", "task-d", "task-b"],
            ],
            "affected_tasks": ["task-b", "task-c", "task-d"],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-detect-cycles"]("complex-cycle-spec")

            assert result["success"] is True
            assert result["data"]["has_cycles"] is True
            assert result["data"]["cycle_count"] == 1

    def test_detect_multiple_independent_cycles(self):
        """Test detection of multiple independent cycles."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [
                ["task-1", "task-2", "task-1"],
                ["task-3", "task-4", "task-5", "task-3"],
            ],
            "affected_tasks": ["task-1", "task-2", "task-3", "task-4", "task-5"],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-detect-cycles"]("multi-cycle-spec")

            assert result["success"] is True
            assert result["data"]["cycle_count"] == 2
            assert len(result["data"]["affected_tasks"]) == 5

    def test_no_cycles_in_acyclic_graph(self):
        """Test no cycles detected in properly structured spec."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [],
            "affected_tasks": [],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-detect-cycles"]("acyclic-spec")

            assert result["success"] is True
            assert result["data"]["has_cycles"] is False
            assert result["data"]["cycles"] == []


class TestPathValidationWorkflows:
    """Integration tests for path validation workflows."""

    def test_validate_existing_paths(self):
        """Test validation of paths that exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "src" / "main.py").touch()
            (Path(tmpdir) / "src" / "utils.py").touch()

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                "valid_paths": ["src/main.py", "src/utils.py"],
                "invalid_paths": [],
            })
            mock_result.stderr = ""

            with patch("subprocess.run", return_value=mock_result):
                from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

                mock_mcp = MagicMock()
                mock_config = MagicMock()
                captured_funcs = {}

                def capture_decorator(mcp, canonical_name):
                    def decorator(func):
                        captured_funcs[canonical_name] = func
                        return func
                    return decorator

                with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                    with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                        register_spec_helper_tools(mock_mcp, mock_config)

                result = captured_funcs["spec-validate-paths"](
                    ["src/main.py", "src/utils.py"],
                    base_directory=tmpdir
                )

                assert result["success"] is True
                assert result["data"]["all_valid"] is True
                assert result["data"]["valid_count"] == 2

    def test_validate_mixed_paths(self):
        """Test validation with mix of valid and invalid paths."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "valid_paths": ["src/existing.py"],
            "invalid_paths": ["src/missing.py", "src/deleted.py"],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-validate-paths"](
                ["src/existing.py", "src/missing.py", "src/deleted.py"]
            )

            assert result["success"] is True
            assert result["data"]["all_valid"] is False
            assert result["data"]["valid_count"] == 1
            assert result["data"]["invalid_count"] == 2

    def test_validate_empty_path_list(self):
        """Test validation with empty path list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "valid_paths": [],
            "invalid_paths": [],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-validate-paths"]([])

            assert result["success"] is True
            assert result["data"]["paths_checked"] == 0
            assert result["data"]["all_valid"] is True


class TestPatternSearchWorkflows:
    """Integration tests for pattern search workflows."""

    def test_find_typescript_files(self):
        """Test finding TypeScript files with glob pattern."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "matches": [
                "src/components/Button.tsx",
                "src/components/Input.tsx",
                "src/utils/helpers.ts",
            ],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-find-patterns"]("*.ts*")

            assert result["success"] is True
            assert result["data"]["total_count"] == 3

    def test_find_test_files(self):
        """Test finding test files with specific pattern."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "matches": [
                "tests/unit/test_auth.py",
                "tests/unit/test_utils.py",
                "tests/integration/test_api.py",
            ],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-find-patterns"]("test_*.py", directory="tests")

            assert result["success"] is True
            assert result["data"]["total_count"] == 3
            assert result["data"]["directory"] == "tests"


class TestRelatedFilesWorkflows:
    """Integration tests for related files discovery workflows."""

    def test_find_files_related_to_source(self):
        """Test finding files related to a source file."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "related_files": [
                {"path": "tests/unit/test_auth.py", "relationship": "test"},
                {"path": "src/utils/crypto.py", "relationship": "import"},
                {"path": "docs/auth.md", "relationship": "documentation"},
            ],
            "spec_references": [
                {"spec_id": "auth-feature-001", "task_id": "task-1-2"},
            ],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-find-related-files"]("src/services/auth.py")

            assert result["success"] is True
            assert result["data"]["total_count"] == 3
            assert len(result["data"]["spec_references"]) == 1

    def test_find_related_files_scoped_to_spec(self):
        """Test finding related files scoped to a specific spec."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "related_files": [
                {"path": "src/models/user.py", "relationship": "import"},
            ],
            "spec_references": [
                {"spec_id": "user-auth-001", "task_id": "task-2-1"},
            ],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_funcs = {}

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    captured_funcs[canonical_name] = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_funcs["spec-find-related-files"](
                "src/services/auth.py",
                spec_id="user-auth-001"
            )

            # Verify command included spec-id
            cmd = mock_run.call_args[0][0]
            assert "--spec-id" in cmd
            assert "user-auth-001" in cmd

            assert result["success"] is True


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
