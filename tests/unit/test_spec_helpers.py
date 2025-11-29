"""
Unit tests for spec helper tools.

Tests the spec-find-related-files tool and associated functionality.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestSpecFindRelatedFiles:
    """Tests for spec-find-related-files tool."""

    def test_successful_find_related_files(self):
        """Test successful find-related-files command execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "related_files": [
                {"path": "src/module/helper.py", "relationship": "import"},
                {"path": "tests/test_module.py", "relationship": "test"},
            ],
            "spec_references": [
                {"spec_id": "feature-123", "task_id": "task-1-1"},
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

            result = captured_funcs["spec-find-related-files"]("src/module/main.py")

            assert result["success"] is True
            assert result["data"]["file_path"] == "src/module/main.py"
            assert len(result["data"]["related_files"]) == 2
            assert result["data"]["total_count"] == 2
            assert len(result["data"]["spec_references"]) == 1

    def test_find_related_files_empty_result(self):
        """Test find-related-files with no related files found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
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

            result = captured_funcs["spec-find-related-files"]("src/unknown/file.py")

            assert result["success"] is True
            assert result["data"]["related_files"] == []
            assert result["data"]["total_count"] == 0

    def test_find_related_files_with_spec_id(self):
        """Test find-related-files with spec_id filter."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"related_files": [], "spec_references": []})
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

            result = captured_funcs["spec-find-related-files"]("src/main.py", spec_id="feature-123")

            # Verify the command included spec_id
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--spec-id" in cmd
            assert "feature-123" in cmd

    def test_find_related_files_with_metadata(self):
        """Test find-related-files with include_metadata=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"related_files": []})
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

            result = captured_funcs["spec-find-related-files"]("src/main.py", include_metadata=True)

            assert result["success"] is True
            assert "metadata" in result["data"]
            assert "command" in result["data"]["metadata"]
            assert "exit_code" in result["data"]["metadata"]

    def test_find_related_files_command_failure(self):
        """Test find-related-files when command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "File not found"

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

            result = captured_funcs["spec-find-related-files"]("nonexistent/file.py")

            assert result["success"] is False
            assert "COMMAND_FAILED" in result["data"]["error_code"]
            assert "File not found" in result["error"]

    def test_find_related_files_timeout(self):
        """Test find-related-files timeout handling."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sdd", 30)):
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

            result = captured_funcs["spec-find-related-files"]("large/project/file.py")

            assert result["success"] is False
            assert "TIMEOUT" in result["data"]["error_code"]

    def test_find_related_files_cli_not_found(self):
        """Test find-related-files when SDD CLI is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError("foundry-cli")):
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

            result = captured_funcs["spec-find-related-files"]("src/main.py")

            assert result["success"] is False
            assert "CLI_NOT_FOUND" in result["data"]["error_code"]


class TestSpecFindPatterns:
    """Tests for spec-find-patterns tool."""

    def test_successful_find_patterns(self):
        """Test successful find-pattern command execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "matches": [
                "src/module/test_helper.py",
                "tests/test_module.py",
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

            result = captured_funcs["spec-find-patterns"]("*.py")

            assert result["success"] is True
            assert result["data"]["pattern"] == "*.py"
            assert len(result["data"]["matches"]) == 2
            assert result["data"]["total_count"] == 2

    def test_find_patterns_empty_result(self):
        """Test find-pattern with no matches found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
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

            result = captured_funcs["spec-find-patterns"]("*.nonexistent")

            assert result["success"] is True
            assert result["data"]["matches"] == []
            assert result["data"]["total_count"] == 0

    def test_find_patterns_with_directory(self):
        """Test find-pattern with directory scope."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"matches": []})
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

            result = captured_funcs["spec-find-patterns"]("*.ts", directory="src/components")

            # Verify the command included directory
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--directory" in cmd
            assert "src/components" in cmd
            assert result["data"]["directory"] == "src/components"

    def test_find_patterns_with_metadata(self):
        """Test find-pattern with include_metadata=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"matches": ["file.py"]})
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

            result = captured_funcs["spec-find-patterns"]("*.py", include_metadata=True)

            assert result["success"] is True
            assert "metadata" in result["data"]
            assert "command" in result["data"]["metadata"]

    def test_find_patterns_command_failure(self):
        """Test find-pattern when command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Invalid pattern"

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

            result = captured_funcs["spec-find-patterns"]("[invalid")

            assert result["success"] is False
            assert "COMMAND_FAILED" in result["data"]["error_code"]

    def test_find_patterns_timeout(self):
        """Test find-pattern timeout handling."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sdd", 30)):
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

            result = captured_funcs["spec-find-patterns"]("**/*")

            assert result["success"] is False
            assert "TIMEOUT" in result["data"]["error_code"]


class TestSpecDetectCycles:
    """Tests for spec-detect-cycles tool."""

    def test_successful_detect_cycles_no_cycles(self):
        """Test successful detect-cycles with no cycles found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [],
            "affected_tasks": [],
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

            result = captured_funcs["spec-detect-cycles"]("feature-123")

            assert result["success"] is True
            assert result["data"]["spec_id"] == "feature-123"
            assert result["data"]["has_cycles"] is False
            assert result["data"]["cycles"] == []
            assert result["data"]["cycle_count"] == 0
            assert result["data"]["affected_tasks"] == []

    def test_successful_detect_cycles_with_cycles(self):
        """Test successful detect-cycles with cycles found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [
                ["task-1", "task-2", "task-3", "task-1"],
                ["task-4", "task-5", "task-4"],
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

            result = captured_funcs["spec-detect-cycles"]("feature-456")

            assert result["success"] is True
            assert result["data"]["has_cycles"] is True
            assert len(result["data"]["cycles"]) == 2
            assert result["data"]["cycle_count"] == 2
            assert len(result["data"]["affected_tasks"]) == 5

    def test_detect_cycles_derives_affected_tasks(self):
        """Test detect-cycles derives affected_tasks when not provided."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "cycles": [["task-a", "task-b", "task-a"]],
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

            result = captured_funcs["spec-detect-cycles"]("feature-789")

            assert result["success"] is True
            # Should derive affected_tasks from cycles
            assert set(result["data"]["affected_tasks"]) == {"task-a", "task-b"}

    def test_detect_cycles_with_metadata(self):
        """Test detect-cycles with include_metadata=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"cycles": []})
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

            result = captured_funcs["spec-detect-cycles"]("feature-123", include_metadata=True)

            assert result["success"] is True
            assert "metadata" in result["data"]
            assert "command" in result["data"]["metadata"]
            assert "exit_code" in result["data"]["metadata"]

    def test_detect_cycles_command_failure(self):
        """Test detect-cycles when command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Spec not found"

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

            result = captured_funcs["spec-detect-cycles"]("nonexistent-spec")

            assert result["success"] is False
            assert "COMMAND_FAILED" in result["data"]["error_code"]
            assert "Spec not found" in result["error"]

    def test_detect_cycles_timeout(self):
        """Test detect-cycles timeout handling."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sdd", 30)):
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

            result = captured_funcs["spec-detect-cycles"]("large-spec")

            assert result["success"] is False
            assert "TIMEOUT" in result["data"]["error_code"]


class TestSpecValidatePaths:
    """Tests for spec-validate-paths tool."""

    def test_successful_validate_paths_all_valid(self):
        """Test successful validate-paths with all paths valid."""
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

            result = captured_funcs["spec-validate-paths"](["src/main.py", "src/utils.py"])

            assert result["success"] is True
            assert result["data"]["paths_checked"] == 2
            assert result["data"]["all_valid"] is True
            assert result["data"]["valid_count"] == 2
            assert result["data"]["invalid_count"] == 0

    def test_successful_validate_paths_some_invalid(self):
        """Test successful validate-paths with some invalid paths."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "valid_paths": ["src/main.py"],
            "invalid_paths": ["src/nonexistent.py"],
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

            result = captured_funcs["spec-validate-paths"](["src/main.py", "src/nonexistent.py"])

            assert result["success"] is True
            assert result["data"]["all_valid"] is False
            assert result["data"]["valid_count"] == 1
            assert result["data"]["invalid_count"] == 1
            assert "src/nonexistent.py" in result["data"]["invalid_paths"]

    def test_validate_paths_with_base_directory(self):
        """Test validate-paths with base_directory option."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"valid_paths": [], "invalid_paths": []})
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

            result = captured_funcs["spec-validate-paths"](["main.py"], base_directory="/project/src")

            # Verify the command included base_directory
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--base-directory" in cmd
            assert "/project/src" in cmd
            assert result["data"]["base_directory"] == "/project/src"

    def test_validate_paths_with_metadata(self):
        """Test validate-paths with include_metadata=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"valid_paths": ["file.py"], "invalid_paths": []})
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

            result = captured_funcs["spec-validate-paths"](["file.py"], include_metadata=True)

            assert result["success"] is True
            assert "metadata" in result["data"]
            assert "command" in result["data"]["metadata"]
            assert "exit_code" in result["data"]["metadata"]

    def test_validate_paths_command_failure(self):
        """Test validate-paths when command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Invalid argument"

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

            result = captured_funcs["spec-validate-paths"](["bad/path"])

            assert result["success"] is False
            assert "COMMAND_FAILED" in result["data"]["error_code"]

    def test_validate_paths_timeout(self):
        """Test validate-paths timeout handling."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sdd", 30)):
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

            result = captured_funcs["spec-validate-paths"](["path1", "path2", "path3"])

            assert result["success"] is False
            assert "TIMEOUT" in result["data"]["error_code"]


class TestResilienceFeatures:
    """Tests for resilience features in spec helper tools."""

    def test_circuit_breaker_error_handling(self):
        """Test that circuit breaker errors are handled gracefully."""
        from foundry_mcp.core.resilience import CircuitBreakerError, CircuitState

        with patch(
            "foundry_mcp.tools.spec_helpers._run_sdd_command",
            side_effect=CircuitBreakerError(
                "SDD CLI circuit breaker is open",
                breaker_name="sdd_cli",
                state=CircuitState.OPEN,
                retry_after=25.0,
            )
        ):
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

            result = captured_funcs["spec-find-related-files"]("src/main.py")

            assert result["success"] is False
            assert "CIRCUIT_OPEN" in result["data"]["error_code"]
            assert "retry" in result["data"]["remediation"].lower()

    def test_timing_metrics_recorded(self):
        """Test that timing metrics are recorded for CLI calls."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import _run_sdd_command, _metrics, _sdd_cli_breaker

            # Reset circuit breaker state
            _sdd_cli_breaker.reset()

            with patch.object(_metrics, "timer") as mock_timer:
                _run_sdd_command(["foundry-cli", "test"], "test_tool")

                # Timer should be called with duration
                mock_timer.assert_called_once()
                call_args = mock_timer.call_args
                assert "test_tool" in call_args[0][0]
                # Duration should be a positive float
                assert call_args[0][1] >= 0


class TestResponseEnvelopeCompliance:
    """Tests to verify response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test success responses include all required fields."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
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

            result = captured_funcs["spec-find-related-files"]("src/main.py")

            # Verify response-v2 envelope
            assert "success" in result
            assert "data" in result
            assert "error" in result
            assert "meta" in result
            assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test error responses include all required fields."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"

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

            result = captured_funcs["spec-find-related-files"]("src/main.py")

            # Verify response-v2 envelope
            assert result["success"] is False
            assert "data" in result
            assert "error_code" in result["data"]
            assert "error_type" in result["data"]
            assert "remediation" in result["data"]
            assert result["error"] is not None
            assert result["meta"]["version"] == "response-v2"
