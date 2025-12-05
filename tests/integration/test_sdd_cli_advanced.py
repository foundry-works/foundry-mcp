"""Integration tests for advanced SDD CLI workflows.

Tests cover:
- Render workflows (basic and enhanced modes)
- Doc-query workflows (find-class, find-function, trace-calls)
- Spec modification workflows (apply, task add/remove)
- Run-tests workflows (presets, discovery)
- LLM doc generation workflows
- Dev utility workflows
"""

import json
import os
import tempfile
from pathlib import Path

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


class TestRenderWorkflows:
    """Tests for render command workflows."""

    def test_render_basic_produces_markdown(self, cli_runner, temp_specs_dir):
        """render basic mode produces valid markdown output."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "render",
                "advanced-test-spec",
                "--mode",
                "basic",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["mode"] == "basic"
        assert "markdown" in data["data"]

    def test_render_with_journal(self, cli_runner, temp_specs_dir):
        """render includes journal entries when requested."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "render",
                "advanced-test-spec",
                "--mode",
                "basic",
                "--include-journal",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_render_enhanced_requires_flag(self, cli_runner, temp_specs_dir):
        """render enhanced mode is gated by feature flag."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "render",
                "advanced-test-spec",
                "--mode",
                "enhanced",
            ],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        # Per response-v2: error is string message, error_code is in data
        assert isinstance(data["error"], str)
        assert data["data"].get("error_code") == "FEATURE_DISABLED"


class TestDocQueryWorkflows:
    """Tests for doc-query command workflows."""

    def test_doc_group_available(self, cli_runner, temp_specs_dir):
        """doc command group is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "doc", "--help"]
        )
        assert result.exit_code == 0
        assert "find-class" in result.output
        assert "find-function" in result.output
        assert "trace-calls" in result.output
        assert "impact" in result.output

    def test_doc_stats_returns_metrics(self, cli_runner, temp_specs_dir):
        """doc stats returns documentation metrics."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "doc", "stats"]
        )
        # May fail if docs not available but should still be valid JSON
        data = json.loads(result.output)
        assert "success" in data


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


class TestLLMDocGenWorkflows:
    """Tests for LLM doc generation workflows."""

    def test_llm_doc_group_available(self, cli_runner, temp_specs_dir):
        """llm-doc command group is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "llm-doc", "--help"]
        )
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "status" in result.output
        assert "cache" in result.output

    def test_llm_doc_status_returns_config(self, cli_runner, temp_specs_dir):
        """llm-doc status returns documentation generation status and artifacts."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "llm-doc", "status"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        # Check for actual response fields
        assert "status" in data["data"]
        assert "output_dir" in data["data"]
        assert "artifacts" in data["data"]


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
            "render",
            "doc",
            "modify",
            "test",
            "llm-doc",
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

    def test_all_top_level_aliases_registered(self, cli_runner, temp_specs_dir):
        """All top-level command aliases are registered."""
        result = cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir), "--help"])
        assert result.exit_code == 0

        # Check key aliases
        expected_aliases = [
            "create",
            "analyze",
            "next-task",
            "render",
            "review-spec",
            "create-pr",
            "run-tests",
            "generate-docs",
        ]
        for alias in expected_aliases:
            assert alias in result.output, f"Missing alias: {alias}"
