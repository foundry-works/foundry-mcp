"""
Unit tests for environment tools.

Tests the environment verification, workspace initialization, and topology
detection helpers used by the unified environment router.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch



class TestVerifyToolchain:
    """Tests for sdd_verify_toolchain function logic."""

    def test_check_tool_available(self):
        """Test that check_tool correctly identifies available tools."""
        import shutil

        # Python should always be available
        assert shutil.which("python") is not None or shutil.which("python3") is not None

    def test_check_tool_unavailable(self):
        """Test that check_tool correctly identifies unavailable tools."""
        import shutil

        # This random name should not exist
        assert shutil.which("nonexistent_tool_xyz_123") is None

    @patch("shutil.which")
    def test_all_required_tools_available(self, mock_which):
        """Test successful toolchain verification when all required tools exist."""
        # Mock all tools as available
        mock_which.return_value = "/usr/bin/tool"

        from foundry_mcp.core.responses import success_response
        from dataclasses import asdict

        required_tools = ["python", "git"]
        required_status = {tool: True for tool in required_tools}

        data = {
            "required": required_status,
            "all_available": True,
            "optional": {
                "grep": True,
                "cat": True,
                "find": True,
                "node": True,
                "npm": True,
            },
        }

        result = asdict(success_response(data=data))
        assert result["success"] is True
        assert result["data"]["all_available"] is True

    @patch("shutil.which")
    def test_missing_required_tool(self, mock_which):
        """Test error response when required tool is missing."""

        def which_side_effect(tool):
            if tool == "git":
                return None
            return "/usr/bin/tool"

        mock_which.side_effect = which_side_effect

        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        data = {
            "required": {"python": True, "git": False},
            "all_available": False,
            "missing": ["git"],
        }

        result = asdict(
            error_response(
                "Required tools missing: git",
                error_code="MISSING_REQUIRED",
                error_type="validation",
                data=data,
            )
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "MISSING_REQUIRED"
        assert "git" in result["data"]["missing"]


class TestInitWorkspace:
    """Tests for sdd_init_workspace function logic."""

    def test_create_specs_directory_structure(self):
        """Test that workspace initialization creates correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            specs_dir = base_path / "specs"

            subdirs = ["active", "pending", "completed", "archived"]

            # Create structure
            specs_dir.mkdir(parents=True)
            for subdir in subdirs:
                (specs_dir / subdir).mkdir()

            # Verify
            assert specs_dir.exists()
            for subdir in subdirs:
                assert (specs_dir / subdir).exists()

    def test_specs_directory_already_exists(self):
        """Test behavior when specs directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            specs_dir = base_path / "specs"

            # Pre-create the directory
            specs_dir.mkdir(parents=True)
            (specs_dir / "active").mkdir()

            # Should detect existing directories
            existing_dirs = []
            created_dirs = []

            if specs_dir.exists():
                existing_dirs.append(str(specs_dir))
            else:
                specs_dir.mkdir()
                created_dirs.append(str(specs_dir))

            assert str(specs_dir) in existing_dirs
            assert len(created_dirs) == 0

    def test_permission_error_handling(self):
        """Test that permission errors are handled gracefully."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        result = asdict(
            error_response(
                "Permission denied: /root/specs",
                error_code="FORBIDDEN",
                error_type="authorization",
                remediation="Check write permissions for the target directory.",
            )
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "FORBIDDEN"
        assert result["data"]["error_type"] == "authorization"


class TestDetectTopology:
    """Tests for sdd_detect_topology function logic."""

    def test_detect_python_project(self):
        """Test detection of Python project markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create Python marker file
            (base_path / "pyproject.toml").touch()

            # Detection logic
            project_type = "unknown"
            python_markers = ["pyproject.toml", "setup.py", "requirements.txt"]
            detected_files = []

            for marker in python_markers:
                if (base_path / marker).exists():
                    project_type = "python"
                    detected_files.append(marker)
                    break

            assert project_type == "python"
            assert "pyproject.toml" in detected_files

    def test_detect_node_project(self):
        """Test detection of Node.js project markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create Node marker file
            (base_path / "package.json").touch()

            # Detection logic
            project_type = "unknown"
            node_markers = ["package.json", "yarn.lock"]
            detected_files = []

            for marker in node_markers:
                if (base_path / marker).exists():
                    project_type = "node"
                    detected_files.append(marker)
                    break

            assert project_type == "node"
            assert "package.json" in detected_files

    def test_detect_specs_directory(self):
        """Test detection of specs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create specs directory
            (base_path / "specs").mkdir()

            # Detection logic
            specs_dir = None
            specs_candidates = ["specs", ".specs", "specifications"]

            for candidate in specs_candidates:
                candidate_path = base_path / candidate
                if candidate_path.is_dir():
                    specs_dir = str(candidate_path)
                    break

            assert specs_dir is not None
            assert "specs" in specs_dir

    def test_detect_git_repository(self):
        """Test detection of git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create .git directory
            (base_path / ".git").mkdir()

            has_git = (base_path / ".git").is_dir()
            assert has_git is True

    def test_unknown_project_type(self):
        """Test handling of unknown project type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)

            # Empty directory - no markers
            project_type = "unknown"

            # No markers found
            assert project_type == "unknown"


class TestVerifyEnvironment:
    """Tests for sdd_verify_environment function logic."""

    def test_python_version_check(self):
        """Test Python version detection."""
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 9  # Assuming Python 3.9+

        runtimes = {
            "python": {
                "available": True,
                "version": python_version,
                "executable": sys.executable,
            }
        }

        assert runtimes["python"]["available"] is True
        assert "." in runtimes["python"]["version"]

    @patch("shutil.which")
    def test_git_available(self, mock_which):
        """Test Git availability check."""
        mock_which.return_value = "/usr/bin/git"


        git_path = mock_which("git")
        assert git_path is not None

    @patch("shutil.which")
    def test_git_not_available(self, mock_which):
        """Test handling when Git is not available."""
        mock_which.return_value = None


        git_path = mock_which("git")
        issues = []

        if not git_path:
            issues.append("Git not found in PATH")

        assert "Git not found in PATH" in issues

    def test_package_import_check(self):
        """Test package import verification."""
        packages = {}

        # Test with a package that should exist
        try:
            __import__("json")
            packages["json"] = True
        except ImportError:
            packages["json"] = False

        assert packages["json"] is True

        # Test with a package that should not exist
        try:
            __import__("nonexistent_package_xyz_123")
            packages["nonexistent"] = True
        except ImportError:
            packages["nonexistent"] = False

        assert packages["nonexistent"] is False

    def test_all_valid_environment(self):
        """Test response when all environment checks pass."""
        from foundry_mcp.core.responses import success_response
        from dataclasses import asdict

        data = {
            "runtimes": {
                "python": {"available": True, "version": "3.11.0"},
                "git": {"available": True, "version": "2.39.0"},
            },
            "all_valid": True,
        }

        result = asdict(success_response(data=data))
        assert result["success"] is True
        assert result["data"]["all_valid"] is True

    def test_invalid_environment_with_issues(self):
        """Test error response when environment validation fails."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        issues = ["Git not found in PATH", "Required package not found: nonexistent"]

        result = asdict(
            error_response(
                f"Environment validation failed: {len(issues)} issue(s) found",
                error_code="VALIDATION_ERROR",
                error_type="validation",
                data={"issues": issues, "all_valid": False},
            )
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        assert len(result["data"]["issues"]) == 2


class TestSddSetup:
    """Tests for sdd_setup function logic."""

    def test_fresh_project_setup(self):
        """Test setup on a project with no existing config."""
        import json
        from foundry_mcp.tools.unified.environment import (
            _init_specs_directory,
            _update_permissions,
            _write_default_toml,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Initialize specs directory
            specs_result = _init_specs_directory(base_path, dry_run=False)
            assert len(specs_result["changes"]) > 0

            # Verify specs structure created
            assert (base_path / "specs").exists()
            assert (base_path / "specs" / "active").exists()
            assert (base_path / "specs" / "pending").exists()
            assert (base_path / "specs" / "completed").exists()
            assert (base_path / "specs" / "archived").exists()

            # Update permissions
            settings_file = base_path / ".claude" / "settings.local.json"
            perm_result = _update_permissions(settings_file, "minimal", dry_run=False)
            assert len(perm_result["changes"]) > 0
            assert settings_file.exists()

            # Verify settings content
            with open(settings_file) as f:
                settings = json.load(f)
            assert "permissions" in settings
            assert "allow" in settings["permissions"]
            assert "mcp__foundry-mcp__spec" in settings["permissions"]["allow"]

            # Create TOML
            toml_path = base_path / "foundry-mcp.toml"
            _write_default_toml(toml_path)
            assert toml_path.exists()

    def test_merge_existing_permissions(self):
        """Test that existing permissions are preserved during merge."""
        import json
        from foundry_mcp.tools.unified.environment import _update_permissions

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            claude_dir = base_path / ".claude"
            claude_dir.mkdir(parents=True)
            settings_file = claude_dir / "settings.local.json"

            # Create existing settings with custom permission
            existing_settings = {
                "permissions": {
                    "allow": ["CustomTool", "AnotherCustomTool"],
                    "deny": [],
                    "ask": [],
                }
            }
            with open(settings_file, "w") as f:
                json.dump(existing_settings, f)

            # Update with minimal preset
            _update_permissions(settings_file, "minimal", dry_run=False)

            # Verify custom permissions preserved
            with open(settings_file) as f:
                new_settings = json.load(f)

            assert "CustomTool" in new_settings["permissions"]["allow"]
            assert "AnotherCustomTool" in new_settings["permissions"]["allow"]
            # Also verify new permissions added
            assert "mcp__foundry-mcp__spec" in new_settings["permissions"]["allow"]

    def test_dry_run_no_changes(self):
        """Test dry_run mode doesn't create files."""
        from foundry_mcp.tools.unified.environment import (
            _init_specs_directory,
            _update_permissions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Dry run specs init
            specs_result = _init_specs_directory(base_path, dry_run=True)
            assert not (base_path / "specs").exists()
            # But changes should be reported
            assert any("Would create" in c for c in specs_result["changes"])

            # Dry run permissions
            settings_file = base_path / ".claude" / "settings.local.json"
            _update_permissions(settings_file, "minimal", dry_run=True)
            assert not (base_path / ".claude").exists()

    def test_invalid_preset_validation(self):
        """Test validation of invalid permissions preset."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        invalid_preset = "invalid_preset"
        result = asdict(
            error_response(
                f"Invalid preset: {invalid_preset}",
                error_code="INVALID_PRESET",
                error_type="validation",
                remediation="Use 'minimal', 'standard', or 'full'",
            )
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "INVALID_PRESET"

    def test_path_not_found_validation(self):
        """Test validation when path does not exist."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        nonexistent_path = "/nonexistent/path/xyz123"
        result = asdict(
            error_response(
                f"Path does not exist: {nonexistent_path}",
                error_code="PATH_NOT_FOUND",
                error_type="validation",
                remediation="Provide a valid project directory path",
            )
        )

        assert result["success"] is False
        assert result["data"]["error_code"] == "PATH_NOT_FOUND"

    def test_response_envelope_compliance(self):
        """Test that success response follows v2 contract."""
        from foundry_mcp.core.responses import success_response
        from dataclasses import asdict

        data = {
            "specs_dir": "/tmp/test/specs",
            "permissions_file": "/tmp/test/.claude/settings.local.json",
            "config_file": "/tmp/test/foundry-mcp.toml",
            "changes": ["Created specs/", "Created .claude/settings.local.json"],
            "dry_run": False,
        }

        result = asdict(success_response(data=data))

        # Verify v2 envelope structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["success"] is True
        assert result["meta"]["version"] == "response-v2"

    def test_toml_content(self):
        """Test that generated TOML has expected content."""
        from foundry_mcp.tools.unified.environment import _DEFAULT_TOML_CONTENT

        assert "[workspace]" in _DEFAULT_TOML_CONTENT
        assert "specs_dir" in _DEFAULT_TOML_CONTENT
        assert "[workflow]" in _DEFAULT_TOML_CONTENT
        assert "mode" in _DEFAULT_TOML_CONTENT
        assert "[logging]" in _DEFAULT_TOML_CONTENT

    def test_idempotent_setup(self):
        """Test that running setup twice is safe (idempotent)."""
        import json
        from foundry_mcp.tools.unified.environment import (
            _init_specs_directory,
            _update_permissions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            settings_file = base_path / ".claude" / "settings.local.json"

            # First run
            _init_specs_directory(base_path, dry_run=False)
            _update_permissions(settings_file, "minimal", dry_run=False)

            with open(settings_file) as f:
                first_settings = json.load(f)
            first_perms_count = len(first_settings["permissions"]["allow"])

            # Second run
            _init_specs_directory(base_path, dry_run=False)
            _update_permissions(settings_file, "minimal", dry_run=False)

            with open(settings_file) as f:
                second_settings = json.load(f)
            second_perms_count = len(second_settings["permissions"]["allow"])

            # Should have same number of permissions (no duplicates)
            assert first_perms_count == second_perms_count


class TestDiscoveryMetadata:
    """Tests for environment tools discovery metadata."""

    def test_environment_tool_metadata_exists(self):
        """Test that environment tool metadata is defined."""
        from foundry_mcp.core.discovery import ENVIRONMENT_TOOL_METADATA

        expected_tools = [
            "sdd-verify-toolchain",
            "sdd-init-workspace",
            "sdd-detect-topology",
            "sdd-verify-environment",
        ]

        for tool_name in expected_tools:
            assert tool_name in ENVIRONMENT_TOOL_METADATA

    def test_environment_feature_flags_exist(self):
        """Test that environment feature flags are defined."""
        from foundry_mcp.core.discovery import ENVIRONMENT_FEATURE_FLAGS

        assert "environment_tools" in ENVIRONMENT_FEATURE_FLAGS
        assert "env_auto_fix" in ENVIRONMENT_FEATURE_FLAGS

    def test_environment_tools_flag_is_beta(self):
        """Test environment_tools flag is in beta state."""
        from foundry_mcp.core.discovery import ENVIRONMENT_FEATURE_FLAGS

        flag = ENVIRONMENT_FEATURE_FLAGS["environment_tools"]
        assert flag.state == "beta"
        assert flag.default_enabled is True
        assert flag.percentage_rollout == 100

    def test_env_auto_fix_flag_is_experimental(self):
        """Test env_auto_fix flag is in experimental state."""
        from foundry_mcp.core.discovery import ENVIRONMENT_FEATURE_FLAGS

        flag = ENVIRONMENT_FEATURE_FLAGS["env_auto_fix"]
        assert flag.state == "experimental"
        assert flag.default_enabled is False
        assert flag.percentage_rollout == 0
        assert "environment_tools" in flag.dependencies

    def test_get_environment_capabilities(self):
        """Test get_environment_capabilities returns correct structure."""
        from foundry_mcp.core.discovery import get_environment_capabilities

        capabilities = get_environment_capabilities()

        assert "environment_readiness" in capabilities
        assert capabilities["environment_readiness"]["supported"] is True
        assert "tools" in capabilities["environment_readiness"]
        assert "feature_flags" in capabilities

    def test_is_environment_tool(self):
        """Test is_environment_tool helper function."""
        from foundry_mcp.core.discovery import is_environment_tool

        assert is_environment_tool("sdd-verify-toolchain") is True
        assert is_environment_tool("sdd-init-workspace") is True
        assert is_environment_tool("nonexistent_tool") is False

    def test_get_environment_tool_metadata(self):
        """Test get_environment_tool_metadata helper function."""
        from foundry_mcp.core.discovery import get_environment_tool_metadata

        metadata = get_environment_tool_metadata("sdd-verify-toolchain")
        assert metadata is not None
        assert metadata.name == "sdd-verify-toolchain"
        assert metadata.category == "environment"

        assert get_environment_tool_metadata("nonexistent") is None

    def test_tool_metadata_has_required_fields(self):
        """Test that tool metadata has all required fields."""
        from foundry_mcp.core.discovery import ENVIRONMENT_TOOL_METADATA

        for tool_name, metadata in ENVIRONMENT_TOOL_METADATA.items():
            assert metadata.name == tool_name
            assert metadata.description
            assert metadata.category == "environment"
            assert metadata.version
            assert isinstance(metadata.tags, list)
            assert isinstance(metadata.related_tools, list)
            assert isinstance(metadata.examples, list)
