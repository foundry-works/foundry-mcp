"""Unit tests for SDD CLI core commands.

Tests cover:
- Validation commands (check, fix, stats, report, analyze-deps)
- Task commands (next, update-status, block, unblock)
- Lifecycle commands (activate, complete, archive, move)
- Session commands (start, status, limits)
"""

import importlib
import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli
from foundry_mcp.cli.commands import testing as testing_commands

tasks_module = importlib.import_module("foundry_mcp.cli.commands.tasks")


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

    # Create a minimal test spec
    test_spec = {
        "id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "status": "active",
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Test Specification",
                "children": ["phase-1"],
                "status": "in_progress",
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "parent": "spec-root",
                "children": ["task-1-1"],
                "status": "in_progress",
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "parent": "phase-1",
                "status": "pending",
                "metadata": {"description": "First task"},
                "dependencies": {},
            },
        },
        "journal": [],
    }

    spec_file = active_dir / "test-spec-001.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir


class TestValidateCheckCommand:
    """Tests for sdd validate check command."""

    def test_validate_check_success(self, cli_runner, temp_specs_dir):
        """validate check returns valid JSON for valid spec."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "check", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "spec_id" in data["data"]
        assert "is_valid" in data["data"]

    def test_validate_check_not_found(self, cli_runner, temp_specs_dir):
        """validate check returns error for non-existent spec."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "check", "nonexistent"],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert "NOT_FOUND" in data["data"]["error_code"]


class TestValidateStatsCommand:
    """Tests for sdd validate stats command."""

    def test_validate_stats_returns_metrics(self, cli_runner, temp_specs_dir):
        """validate stats returns specification metrics."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "stats", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "title" in data["data"]
        assert "version" in data["data"]
        assert "totals" in data["data"]


class TestValidateReportCommand:
    """Tests for sdd validate report command."""

    def test_validate_report_all_sections(self, cli_runner, temp_specs_dir):
        """validate report includes all sections by default."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "report", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        sections = set(data["data"]["sections"])
        assert "validation" in sections
        assert "stats" in sections
        assert "health" in sections

    def test_validate_report_specific_sections(self, cli_runner, temp_specs_dir):
        """validate report filters to requested sections."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "validate",
                "report",
                "--sections",
                "health",
                "test-spec-001",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "health" in data["data"]


class TestValidateAnalyzeDepsCommand:
    """Tests for sdd validate analyze-deps command."""

    def test_analyze_deps_returns_dependency_info(self, cli_runner, temp_specs_dir):
        """analyze-deps returns dependency analysis."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "validate",
                "analyze-deps",
                "test-spec-001",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "dependency_count" in data["data"]
        assert "bottlenecks" in data["data"]
        assert "circular_deps" in data["data"]
        assert "has_circular_deps" in data["data"]

    def test_analyze_deps_with_threshold(self, cli_runner, temp_specs_dir):
        """analyze-deps respects bottleneck threshold option."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "validate",
                "analyze-deps",
                "--bottleneck-threshold",
                "5",
                "test-spec-001",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["data"]["bottleneck_threshold"] == 5


class TestTasksNextCommand:
    """Tests for sdd tasks next command."""

    def test_tasks_next_finds_task(self, cli_runner, temp_specs_dir):
        """tasks next finds the next actionable task."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "tasks", "next", "test-spec-001"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        # Should find task-1-1 as it's pending and unblocked
        assert data["data"]["found"] is True or data["data"].get("task_id") is not None

    def test_tasks_next_emits_error_when_spec_missing(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """tasks next exits immediately when spec cannot be loaded."""
        monkeypatch.setattr(tasks_module, "load_spec", lambda *_: None)
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "tasks", "next", "missing-spec"]
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "SPEC_NOT_FOUND"


class TestSessionCommands:
    """Tests for sdd session commands."""

    def test_session_start(self, cli_runner, temp_specs_dir):
        """session start creates a new session."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "start"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "session_id" in data["data"]

    def test_session_status(self, cli_runner, temp_specs_dir):
        """session status returns current session info."""
        # Start a session first
        cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir), "session", "start"])
        # Then check status
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "status"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_session_limits(self, cli_runner, temp_specs_dir):
        """session limits returns limit configuration."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "limits"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_session_work_mode_default(self, cli_runner, temp_specs_dir):
        """session work-mode returns default mode when env not set."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "work-mode"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["work_mode"] == "single"  # Default mode
        assert "modes_available" in data["data"]
        assert "single" in data["data"]["modes_available"]
        assert "autonomous" in data["data"]["modes_available"]

    def test_session_work_mode_with_env(self, cli_runner, temp_specs_dir, monkeypatch):
        """session work-mode respects FOUNDRY_MCP_WORK_MODE env var."""
        monkeypatch.setenv("FOUNDRY_MCP_WORK_MODE", "autonomous")
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "work-mode"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["work_mode"] == "autonomous"

    def test_session_capabilities(self, cli_runner, temp_specs_dir):
        """session capabilities returns CLI capability manifest."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "capabilities"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "capabilities" in data["data"]
        assert data["data"]["capabilities"]["json_output"] is True
        assert "command_groups" in data["data"]
        assert "feature_flags" in data["data"]

    def test_session_record(self, cli_runner, temp_specs_dir):
        """session record logs consultation usage."""
        # Start a session first
        cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir), "session", "start"])
        # Record a consultation
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "session",
                "record",
                "--tokens",
                "1000",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_session_reset(self, cli_runner, temp_specs_dir):
        """session reset clears session state."""
        # Start a session first
        cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir), "session", "start"])
        # Reset it
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "reset"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "message" in data["data"]

    def test_session_token_usage_without_agent(self, cli_runner, temp_specs_dir):
        """session token-usage returns unavailable without claude-code agent."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "token-usage"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["available"] is False
        assert "claude-code" in data["data"]["reason"]

    def test_session_token_usage_with_agent(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """session token-usage works with claude-code agent type."""
        monkeypatch.setenv("FOUNDRY_MCP_AGENT_TYPE", "claude-code")
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "token-usage"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["available"] is True
        assert data["data"]["agent_type"] == "claude-code"

    def test_session_generate_marker_without_agent(self, cli_runner, temp_specs_dir):
        """session generate-marker returns unavailable without claude-code agent."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "generate-marker"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["available"] is False
        assert "claude-code" in data["data"]["reason"]

    def test_session_generate_marker_with_agent(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """session generate-marker creates marker with claude-code agent."""
        monkeypatch.setenv("FOUNDRY_MCP_AGENT_TYPE", "claude-code")
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "generate-marker"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "marker" in data["data"]
        assert data["data"]["marker"].startswith("SESSION_MARKER_")


class TestCacheCommands:
    """Tests for foundry-cli cache commands."""

    def test_cache_info(self, cli_runner, temp_specs_dir):
        """cache info returns cache statistics."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "cache", "info"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        # Either enabled with stats or disabled with message
        assert "enabled" in data["data"]

    def test_cache_info_disabled(self, cli_runner, temp_specs_dir, monkeypatch):
        """cache info handles disabled cache."""
        monkeypatch.setenv("FOUNDRY_MCP_CACHE_DISABLED", "1")
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "cache", "info"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["enabled"] is False

    def test_cache_clear(self, cli_runner, temp_specs_dir):
        """cache clear removes entries."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "cache", "clear"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        # Either entries_deleted count or disabled status
        assert "entries_deleted" in data["data"] or data["data"].get("enabled") is False

    def test_cache_clear_with_spec_filter(self, cli_runner, temp_specs_dir):
        """cache clear accepts spec-id filter."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "cache",
                "clear",
                "--spec-id",
                "test-spec",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_cache_clear_with_review_type_filter(self, cli_runner, temp_specs_dir):
        """cache clear accepts review-type filter."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "cache",
                "clear",
                "--review-type",
                "fidelity",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True

    def test_cache_cleanup(self, cli_runner, temp_specs_dir):
        """cache cleanup removes expired entries."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "cache", "cleanup"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        # Either entries_removed count or disabled status
        assert "entries_removed" in data["data"] or data["data"].get("enabled") is False


class TestSchemaCommands:
    """Tests for foundry-cli specs schema command."""

    def test_specs_schema(self, cli_runner, temp_specs_dir):
        """specs schema returns valid JSON schema."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "specs", "schema"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "schema" in data["data"]
        assert "version" in data["data"]
        # Schema should have standard JSON Schema fields
        schema = data["data"]["schema"]
        assert "$schema" in schema or "type" in schema

    def test_specs_schema_has_required_properties(self, cli_runner, temp_specs_dir):
        """specs schema includes required spec properties."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "specs", "schema"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        schema = data["data"]["schema"]
        # Schema should define spec_id and hierarchy properties
        properties = schema.get("properties", {})
        assert "spec_id" in properties or "hierarchy" in properties


class TestLifecycleCommands:
    """Tests for sdd lifecycle commands."""

    def test_lifecycle_state(self, cli_runner, temp_specs_dir):
        """lifecycle state returns spec lifecycle info."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "lifecycle", "state", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "folder" in data["data"]


class TestPRCommands:
    """Tests for sdd pr commands."""

    def test_pr_status_checks_prerequisites(self, cli_runner, temp_specs_dir):
        """pr status returns prerequisite checks."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "pr", "status"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "prerequisites" in data["data"]
        assert "github_cli" in data["data"]["prerequisites"]
        assert "git" in data["data"]["prerequisites"]
        assert "llm" in data["data"]["prerequisites"]

    def test_pr_context_returns_spec_info(self, cli_runner, temp_specs_dir):
        """pr context returns spec context for PR description."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "pr", "context", "test-spec-001"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["spec_id"] == "test-spec-001"
        assert "progress" in data["data"]
        assert "completed_tasks" in data["data"]


class TestReviewCommands:
    """Tests for sdd review commands."""

    def test_review_tools_lists_tools(self, cli_runner, temp_specs_dir):
        """review tools returns tool list with availability."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "review", "tools"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "tools" in data["data"]
        assert "llm_status" in data["data"]
        assert "review_types" in data["data"]

    def test_review_plan_tools_lists_toolchains(self, cli_runner, temp_specs_dir):
        """review plan-tools returns plan toolchains with status."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "review", "plan-tools"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "plan_tools" in data["data"]
        assert "recommendations" in data["data"]
        # Check that plan tools have status
        for tool in data["data"]["plan_tools"]:
            assert "status" in tool
            assert "name" in tool

    def test_review_spec_quick_returns_structural_findings(
        self, cli_runner, temp_specs_dir
    ):
        """review spec quick returns native structural findings."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "review",
                "spec",
                "test-spec-001",
                "--type",
                "quick",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["review_type"] == "quick"
        assert "findings" in data["data"]
        assert "progress" in data["data"]


class TestModifyCommands:
    """Tests for sdd modify commands."""

    def test_modify_group_exists(self, cli_runner, temp_specs_dir):
        """modify group is available."""
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

    def test_modify_task_subgroup_exists(self, cli_runner, temp_specs_dir):
        """modify task subgroup is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "modify", "task", "--help"]
        )
        assert result.exit_code == 0
        assert "add" in result.output
        assert "remove" in result.output

    def test_modify_phase_subgroup_exists(self, cli_runner, temp_specs_dir):
        """modify phase subgroup is available."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "modify", "phase", "--help"]
        )
        assert result.exit_code == 0
        assert "add" in result.output


class TestTestingCommands:
    """Tests for sdd test commands."""

    def test_test_group_exists(self, cli_runner, temp_specs_dir):
        """test group is available with expected subcommands."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "--help"]
        )
        assert result.exit_code == 0
        assert "run" in result.output
        assert "discover" in result.output
        assert "presets" in result.output
        assert "check-tools" in result.output
        assert "quick" in result.output
        assert "unit" in result.output

    def test_test_presets_returns_preset_list(self, cli_runner, temp_specs_dir):
        """test presets returns available presets."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "presets"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "presets" in data["data"]
        assert "quick" in data["data"]["presets"]
        assert "full" in data["data"]["presets"]
        assert "unit" in data["data"]["presets"]

    def test_test_check_tools_returns_status(self, cli_runner, temp_specs_dir):
        """test check-tools returns toolchain status."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "check-tools"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "tools" in data["data"]
        assert "pytest" in data["data"]["tools"]

    def test_test_run_success(self, cli_runner, temp_specs_dir, monkeypatch):
        """test run emits success payload when pytest passes."""
        monkeypatch.setattr(
            testing_commands.subprocess,
            "run",
            lambda *_, **__: SimpleNamespace(
                returncode=0,
                stdout="1 passed in 0.12s",
                stderr="",
            ),
        )

        def no_timeout(seconds, message):
            def decorator(func):
                return func

            return decorator

        monkeypatch.setattr(testing_commands, "with_sync_timeout", no_timeout)

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "test",
                "run",
                "--timeout",
                "15",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["passed"] is True
        assert data["data"]["summary"]["passed"] == 1

    def test_test_run_failure_emits_error(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """test run returns error envelope when pytest fails."""
        monkeypatch.setattr(
            testing_commands.subprocess,
            "run",
            lambda *_, **__: SimpleNamespace(
                returncode=1,
                stdout="1 failed, 0 passed",
                stderr="E   AssertionError",
            ),
        )

        def no_timeout(seconds, message):
            def decorator(func):
                return func

            return decorator

        monkeypatch.setattr(testing_commands, "with_sync_timeout", no_timeout)

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "test",
                "run",
                "--timeout",
                "20",
            ],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "TEST_FAILED"

    def test_test_run_honors_timeout_flag(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """test run uses provided timeout for signal wrapper."""
        monkeypatch.setattr(
            testing_commands.subprocess,
            "run",
            lambda *_, **__: SimpleNamespace(
                returncode=0,
                stdout="1 passed",
                stderr="",
            ),
        )

        captured = {}

        def capture_timeout(seconds, message):
            captured["seconds"] = seconds
            captured["message"] = message

            def decorator(func):
                return func

            return decorator

        monkeypatch.setattr(testing_commands, "with_sync_timeout", capture_timeout)

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "test",
                "run",
                "--timeout",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["seconds"] == 42
        assert "42" in captured["message"]


class TestDevCommands:
    """Tests for sdd dev commands."""

    def test_dev_group_exists(self, cli_runner, temp_specs_dir):
        """dev group is available with expected subcommands."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "dev", "--help"]
        )
        assert result.exit_code == 0
        assert "gendocs" in result.output
        assert "install" in result.output
        assert "start" in result.output
        assert "check" in result.output

    def test_dev_check_returns_tools(self, cli_runner, temp_specs_dir):
        """dev check returns development tool status."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "dev", "check"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"] is True
        assert "tools" in data["data"]
        assert "python" in data["data"]["tools"]
        assert "pip" in data["data"]["tools"]
        assert "git" in data["data"]["tools"]


class TestCLIJsonOutput:
    """Tests verifying all commands output valid JSON."""

    def test_all_commands_output_json(self, cli_runner, temp_specs_dir):
        """All commands should output valid JSON."""
        commands = [
            ["validate", "check", "test-spec-001"],
            ["validate", "stats", "test-spec-001"],
            ["validate", "report", "test-spec-001"],
            ["validate", "analyze-deps", "test-spec-001"],
            ["tasks", "next", "test-spec-001"],
            ["session", "status"],
            ["session", "limits"],
        ]

        for cmd in commands:
            result = cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir)] + cmd)
            # Should be valid JSON regardless of success/failure
            try:
                data = json.loads(result.output)
                assert "success" in data
            except json.JSONDecodeError:
                pytest.fail(
                    f"Command {cmd} did not output valid JSON: {result.output[:200]}"
                )


class TestTasksCompleteCommand:
    """Tests for sdd tasks complete command."""

    def test_tasks_complete_updates_status(self, cli_runner, temp_specs_dir):
        """tasks complete marks a task as completed and journals the note."""
        spec_path = temp_specs_dir / "active" / "test-spec-001.json"
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "tasks",
                "complete",
                "test-spec-001",
                "task-1-1",
                "--note",
                "Finished wiring",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["status"] == "completed"

        updated_spec = json.loads(spec_path.read_text())
        assert updated_spec["hierarchy"]["task-1-1"]["status"] == "completed"
        assert updated_spec["journal"], "journal entry should be created"

    def test_tasks_complete_records_previous_status(self, cli_runner, temp_specs_dir):
        """tasks complete stores the task's prior status in journal metadata."""
        spec_path = temp_specs_dir / "active" / "test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        spec_data["hierarchy"]["task-1-1"]["status"] = "in_progress"
        spec_path.write_text(json.dumps(spec_data))

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "tasks",
                "complete",
                "test-spec-001",
                "task-1-1",
                "--note",
                "Wrapped up work",
            ],
        )
        assert result.exit_code == 0, result.output
        updated_spec = json.loads(spec_path.read_text())
        entry = updated_spec["journal"][-1]
        assert entry["metadata"].get("previous_status") == "in_progress"


class TestTestDiscoverCommand:
    """Tests for sdd test discover command."""

    def test_test_discover_applies_pattern(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """--pattern should add a pytest -k expression to the collect command."""
        calls = []

        def fake_run(cmd, capture_output, text, timeout, cwd):  # pylint: disable=unused-argument
            calls.append(cmd)
            return SimpleNamespace(
                returncode=0, stdout="tests/test_sample.py::test_ok\n", stderr=""
            )

        monkeypatch.setattr(testing_commands.subprocess, "run", fake_run)

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "test",
                "discover",
                "--pattern",
                "unit",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["pattern"] == "unit"
        assert ["-k", "unit"] in [calls[0][i : i + 2] for i in range(len(calls[0]) - 1)]

    def test_test_discover_runs_tests_when_no_list(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """--no-list should execute pytest after discovery."""
        calls = []

        def fake_run(cmd, capture_output, text, timeout, cwd):  # pylint: disable=unused-argument
            calls.append(cmd)
            if len(calls) == 1:
                return SimpleNamespace(
                    returncode=0, stdout="tests/test_sample.py::test_ok\n", stderr=""
                )
            return SimpleNamespace(
                returncode=0, stdout="1 passed in 0.01s\n", stderr=""
            )

        monkeypatch.setattr(testing_commands.subprocess, "run", fake_run)

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "test",
                "discover",
                "--no-list",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["list_only"] is False
        assert data["data"]["test_run"]["passed"] is True
        assert len(calls) == 2

    def test_test_discover_propagates_test_failures(
        self, cli_runner, temp_specs_dir, monkeypatch
    ):
        """Failing test execution should return an error code."""
        calls = []

        def fake_run(cmd, capture_output, text, timeout, cwd):  # pylint: disable=unused-argument
            calls.append(cmd)
            if len(calls) == 1:
                return SimpleNamespace(
                    returncode=0, stdout="tests/test_sample.py::test_ok\n", stderr=""
                )
            return SimpleNamespace(
                returncode=1, stdout="", stderr="FAILED test_sample.py::test_ok"
            )

        monkeypatch.setattr(testing_commands.subprocess, "run", fake_run)

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "test",
                "discover",
                "--no-list",
            ],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "TEST_FAILED"
