"""Integration tests for the structured CLI groups.

Covers:
- `modify` commands
- `test` commands
- `dev` commands
- Cross-group registration
"""

import json

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_specs_dir(tmp_path):
    """Create a temporary specs directory with test specs."""
    specs_dir = tmp_path / "specs"
    active_dir = specs_dir / "active"
    active_dir.mkdir(parents=True)

    # Create a comprehensive test spec
    test_spec = {
        "id": "advanced-test-spec",
        "title": "Advanced Test Specification",
        "version": "1.0.0",
        "status": "active",
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Advanced Test Specification",
                "children": ["phase-1"],
                "status": "in_progress",
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1: Implementation",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "status": "in_progress",
            },
            "task-1-1": {
                "type": "task",
                "title": "Implement core feature",
                "parent": "phase-1",
                "status": "completed",
                "metadata": {
                    "description": "Core feature implementation",
                    "file_path": "src/core.py",
                },
                "dependencies": {},
            },
            "task-1-2": {
                "type": "task",
                "title": "Add tests",
                "parent": "phase-1",
                "status": "pending",
                "metadata": {"description": "Test implementation"},
                "dependencies": {"task-1-1": "completed"},
            },
        },
        "journal": [
            {
                "timestamp": "2025-01-01T00:00:00Z",
                "type": "note",
                "title": "Started implementation",
                "content": "Beginning core feature work",
            }
        ],
    }

    spec_file = active_dir / "advanced-test-spec.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir


class TestModifyWorkflows:
    """Tests for spec modification workflows."""

    def test_modify_group_available(self, cli_runner, temp_specs_dir):
        """modify command group is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "modify", "--help"]
        )
        assert result.exit_code == 0
        assert "apply" in result.output
        assert "task" in result.output
        assert "phase" in result.output
        assert "assumption" in result.output
        assert "revision" in result.output
        assert "frontmatter" in result.output

    def test_modify_task_subgroup_available(self, cli_runner, temp_specs_dir):
        """modify task subgroup is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "modify", "task", "--help"]
        )
        assert result.exit_code == 0
        assert "add" in result.output
        assert "remove" in result.output

    def test_modify_phase_subgroup_available(self, cli_runner, temp_specs_dir):
        """modify phase subgroup is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "modify", "phase", "--help"]
        )
        assert result.exit_code == 0
        assert "add" in result.output


class TestRunTestsWorkflows:
    """Tests for run-tests command workflows."""

    def test_test_group_available(self, cli_runner, temp_specs_dir):
        """test command group is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "--help"]
        )
        assert result.exit_code == 0
        assert "run" in result.output
        assert "discover" in result.output
        assert "presets" in result.output
        assert "check-tools" in result.output

    def test_test_presets_lists_all(self, cli_runner, temp_specs_dir):
        """test presets returns all preset configurations."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "presets"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        presets = data["data"]["presets"]
        assert "quick" in presets
        assert "full" in presets
        assert "unit" in presets
        assert "integration" in presets
        assert "smoke" in presets

    def test_test_check_tools_returns_status(self, cli_runner, temp_specs_dir):
        """test check-tools returns tool availability."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "check-tools"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "tools" in data["data"]
        # pytest should be available in test environment
        assert data["data"]["tools"]["pytest"]["available"] is True


class TestDevWorkflows:
    """Tests for dev utility workflows."""

    def test_dev_group_available(self, cli_runner, temp_specs_dir):
        """dev command group is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "dev", "--help"]
        )
        assert result.exit_code == 0
        assert "gendocs" in result.output
        assert "install" in result.output
        assert "start" in result.output
        assert "check" in result.output

    def test_dev_check_returns_env_status(self, cli_runner, temp_specs_dir):
        """dev check returns environment status."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "dev", "check"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "tools" in data["data"]
        # Python should definitely be available
        assert data["data"]["tools"]["python"]["available"] is True


class TestCrossWorkflowIntegration:
    """Tests for cross-workflow integration."""

    def test_all_command_groups_registered(self, cli_runner, temp_specs_dir):
        """All advanced command groups are registered."""
        result = cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir), "--help"])
        assert result.exit_code == 0

        # Check all groups are present
        expected_groups = [
            "modify",
            "test",
            "dev",
            "validate",
            "review",
            "pr",
            "specs",
            "tasks",
            "lifecycle",
            "session",
        ]
        for group in expected_groups:
            assert group in result.output, f"Missing command group: {group}"
