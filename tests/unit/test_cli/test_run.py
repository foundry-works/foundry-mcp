"""Unit tests for the foundry run CLI command."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def temp_specs_dir(tmp_path):
    """Create a temporary specs directory with a resolvable spec."""
    specs_dir = tmp_path / "specs"
    active_dir = specs_dir / "active"
    active_dir.mkdir(parents=True)

    test_spec = {
        "id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "status": "active",
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Test",
                "children": [],
                "status": "in_progress",
            }
        },
        "journal": [],
    }
    spec_file = active_dir / "test-spec-001.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir


class TestRunTmuxMissing:
    """Tests for error when tmux is not installed."""

    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value=None)
    def test_tmux_not_found_shows_error(
        self, mock_which, cli_runner, temp_specs_dir
    ):
        """run returns clear error when tmux is not installed."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "DEPENDENCY_MISSING"
        assert "tmux" in data["error"]

    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value=None)
    def test_tmux_not_found_provides_remediation(
        self, mock_which, cli_runner, temp_specs_dir
    ):
        """Error includes install instructions."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        data = json.loads(result.output)
        assert "install tmux" in data["data"].get("remediation", "").lower()


class TestRunSpecNotFound:
    """Tests for error when spec does not exist."""

    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_error_when_spec_not_found(
        self, mock_which, cli_runner, temp_specs_dir
    ):
        """run returns error for non-existent spec."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "nonexistent-spec"],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "NOT_FOUND"


class TestRunDefaultOptions:
    """Tests for correct tmux command construction with defaults."""

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_creates_tmux_session_with_agent_command(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run creates tmux session with correct agent command."""
        # has-session returns 1 (no existing session)
        # new-session, split-window, select-pane succeed
        # attach succeeds
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),  # has-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # new-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # split-window
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # select-pane
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # attach
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        # Verify new-session call
        new_session_call = mock_run.call_args_list[1]
        cmd_args = new_session_call[0][0]
        assert cmd_args[0] == "tmux"
        assert cmd_args[1] == "new-session"
        assert "-d" in cmd_args
        assert "-s" in cmd_args
        # Agent command should include env vars and claude -p
        agent_cmd = cmd_args[-1]
        assert "FOUNDRY_MCP_ROLE=autonomy_runner" in agent_cmd
        assert "FOUNDRY_MCP_AUTONOMY_POSTURE=unattended" in agent_cmd
        assert "claude -p" in agent_cmd
        assert "test-spec-001" in agent_cmd

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_creates_watcher_pane_with_delay(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run creates watcher pane with sleep delay."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),  # has-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # new-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # split-window
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # select-pane
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # attach
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        # Verify split-window call contains watcher command
        split_call = mock_run.call_args_list[2]
        cmd_args = split_call[0][0]
        assert cmd_args[0] == "tmux"
        assert cmd_args[1] == "split-window"
        watcher_cmd = cmd_args[-1]
        assert "sleep 5" in watcher_cmd
        assert "foundry watch test-spec-001" in watcher_cmd

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_default_layout_is_vertical(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """Default layout uses vertical split (-v flag)."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),  # has-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # new-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # split-window
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # select-pane
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # attach
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        split_call = mock_run.call_args_list[2]
        cmd_args = split_call[0][0]
        assert "-v" in cmd_args

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_selects_agent_pane_after_split(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run selects the agent pane (pane 0) after splitting."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        select_call = mock_run.call_args_list[3]
        cmd_args = select_call[0][0]
        assert cmd_args[0] == "tmux"
        assert cmd_args[1] == "select-pane"
        # Should target pane 0.0
        pane_target = cmd_args[3]
        assert pane_target.endswith(":0.0")

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    @patch.dict("os.environ", {}, clear=False)
    def test_attaches_by_default(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run attaches to tmux session by default (no --detach, not in tmux)."""
        # Ensure TMUX is not set
        import os
        os.environ.pop("TMUX", None)

        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # attach
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        attach_call = mock_run.call_args_list[4]
        cmd_args = attach_call[0][0]
        assert cmd_args == ["tmux", "attach", "-t", "foundry-test-spec-001"]


class TestRunInsideTmux:
    """Tests for behavior when already inside a tmux session."""

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    @patch.dict("os.environ", {"TMUX": "/tmp/tmux-1000/default,12345,0"})
    def test_uses_switch_client_when_in_tmux(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run uses switch-client instead of attach when already in tmux."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),  # has-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # new-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # split-window
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # select-pane
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # switch-client
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        switch_call = mock_run.call_args_list[4]
        cmd_args = switch_call[0][0]
        assert cmd_args == ["tmux", "switch-client", "-t", "foundry-test-spec-001"]

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    @patch.dict("os.environ", {"TMUX": "/tmp/tmux-1000/default,12345,0"})
    def test_action_is_switching_when_in_tmux(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run emits action='switching' when already in tmux."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["action"] == "switching"
        assert "switch-client" in data["data"]["reattach_command"]

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    @patch.dict("os.environ", {"TMUX": "/tmp/tmux-1000/default,12345,0"})
    def test_detach_ignores_tmux_env(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """--detach still detaches normally even when inside tmux."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "--detach", "test-spec-001"],
        )

        assert result.exit_code == 0
        assert mock_run.call_count == 4
        data = json.loads(result.output)
        assert data["data"]["action"] == "detached"


class TestRunDetachMode:
    """Tests for --detach flag behavior."""

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_detach_skips_attach(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """--detach creates session but does not attach."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),  # has-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # new-session
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # split-window
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # select-pane
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "--detach", "test-spec-001"],
        )

        assert result.exit_code == 0
        # Only 4 subprocess calls (no attach)
        assert mock_run.call_count == 4

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_detach_shows_reattach_command(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """--detach emits reattach command in response."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "-d", "test-spec-001"],
        )

        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["action"] == "detached"
        assert "tmux attach" in data["data"]["reattach_command"]


class TestRunLayoutOption:
    """Tests for --layout flag."""

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_horizontal_layout_uses_h_flag(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """--layout horizontal uses -h split flag."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir", str(temp_specs_dir),
                "run", "--layout", "horizontal", "--detach", "test-spec-001",
            ],
        )

        split_call = mock_run.call_args_list[2]
        cmd_args = split_call[0][0]
        assert "-h" in cmd_args
        assert "-v" not in cmd_args

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_vertical_layout_uses_v_flag(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """--layout vertical uses -v split flag."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir", str(temp_specs_dir),
                "run", "--layout", "vertical", "--detach", "test-spec-001",
            ],
        )

        split_call = mock_run.call_args_list[2]
        cmd_args = split_call[0][0]
        assert "-v" in cmd_args


class TestRunPostureOption:
    """Tests for --posture flag."""

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_supervised_posture_sets_env_var(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """--posture supervised passes correct env var to agent command."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir", str(temp_specs_dir),
                "run", "--posture", "supervised", "--detach", "test-spec-001",
            ],
        )

        new_session_call = mock_run.call_args_list[1]
        agent_cmd = new_session_call[0][0][-1]
        assert "FOUNDRY_MCP_AUTONOMY_POSTURE=supervised" in agent_cmd

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_default_posture_is_unattended(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """Default posture is unattended."""
        mock_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "--detach", "test-spec-001"],
        )

        new_session_call = mock_run.call_args_list[1]
        agent_cmd = new_session_call[0][0][-1]
        assert "FOUNDRY_MCP_AUTONOMY_POSTURE=unattended" in agent_cmd


class TestRunSessionExists:
    """Tests for error when tmux session already exists."""

    @patch("foundry_mcp.cli.commands.run.subprocess.run")
    @patch("foundry_mcp.cli.commands.run.shutil.which", return_value="/usr/bin/tmux")
    def test_error_when_session_exists(
        self, mock_which, mock_run, cli_runner, temp_specs_dir
    ):
        """run returns error if tmux session already exists."""
        # has-session returns 0 (session exists)
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "run", "test-spec-001"],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "ALREADY_EXISTS"
