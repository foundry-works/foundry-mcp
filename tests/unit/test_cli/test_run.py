"""Unit tests for the foundry run CLI command."""

import json
from unittest.mock import patch

import pytest

from foundry_mcp.cli.main import cli


class TestRunClaudeMissing:
    """Tests for error when claude CLI is not installed."""

    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value=None)
    def test_claude_not_found_shows_error(self, mock_which, cli_runner, temp_specs_dir):
        """run returns clear error when claude CLI is not installed."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "DEPENDENCY_MISSING"
        assert "claude" in data["error"]

    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value=None)
    def test_claude_not_found_provides_remediation(self, mock_which, cli_runner, temp_specs_dir):
        """Error includes install instructions."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        data = json.loads(result.output)
        assert "claude" in data["data"].get("remediation", "").lower()


class TestRunSpecNotFound:
    """Tests for error when spec does not exist."""

    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/claude")
    def test_error_when_spec_not_found(self, mock_which, cli_runner, temp_specs_dir):
        """run returns error for non-existent spec."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "nonexistent-spec"],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "NOT_FOUND"


class TestRunExecvpe:
    """Tests for correct command construction when launching claude."""

    @patch("foundry_mcp.cli.commands.run.os.execvpe")
    @patch("foundry_mcp.cli.commands.run.signal.alarm")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/claude")
    def test_launches_claude_with_correct_args(self, mock_which, mock_alarm, mock_execvpe, cli_runner, temp_specs_dir):
        """run launches claude with the foundry-implement-auto skill."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        mock_execvpe.assert_called_once()
        args = mock_execvpe.call_args
        cmd = args[0][1]  # The command list
        assert cmd[0] == "/usr/bin/claude"
        assert "/foundry-implement-auto test-spec-001" in cmd[1]

    @patch("foundry_mcp.cli.commands.run.os.execvpe")
    @patch("foundry_mcp.cli.commands.run.signal.alarm")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/claude")
    def test_unattended_includes_skip_permissions(self, mock_which, mock_alarm, mock_execvpe, cli_runner, temp_specs_dir):
        """Default unattended posture includes --dangerously-skip-permissions."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        mock_execvpe.assert_called_once()
        cmd = mock_execvpe.call_args[0][1]
        assert "--dangerously-skip-permissions" in cmd

    @patch("foundry_mcp.cli.commands.run.os.execvpe")
    @patch("foundry_mcp.cli.commands.run.signal.alarm")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/claude")
    def test_supervised_omits_skip_permissions(self, mock_which, mock_alarm, mock_execvpe, cli_runner, temp_specs_dir):
        """--posture supervised does NOT include --dangerously-skip-permissions."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "--posture", "supervised", "test-spec-001"],
        )

        mock_execvpe.assert_called_once()
        cmd = mock_execvpe.call_args[0][1]
        assert "--dangerously-skip-permissions" not in cmd

    @patch("foundry_mcp.cli.commands.run.os.execvpe")
    @patch("foundry_mcp.cli.commands.run.signal.alarm")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/claude")
    def test_sets_environment_variables(self, mock_which, mock_alarm, mock_execvpe, cli_runner, temp_specs_dir):
        """run sets FOUNDRY_MCP_ROLE and FOUNDRY_MCP_AUTONOMY_POSTURE env vars."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        mock_execvpe.assert_called_once()
        env = mock_execvpe.call_args[0][2]
        assert env["FOUNDRY_MCP_ROLE"] == "autonomy_runner"
        assert env["FOUNDRY_MCP_AUTONOMY_POSTURE"] == "unattended"

    @patch("foundry_mcp.cli.commands.run.os.execvpe")
    @patch("foundry_mcp.cli.commands.run.signal.alarm")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/claude")
    def test_cancels_alarm_before_exec(self, mock_which, mock_alarm, mock_execvpe, cli_runner, temp_specs_dir):
        """run cancels the SIGALRM timeout before launching claude."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        mock_alarm.assert_called_with(0)
