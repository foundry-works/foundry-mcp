"""Unit tests for the foundry stop CLI command.

Tests cover:
- Signal file creation with correct JSON payload
- Error when spec not found
- --force mode process kill (mocked subprocess/os.kill)
- --wait mode poll and exit on terminal status (mocked AutonomyStorage)
- --wait mode timeout
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.cli.main import cli


class TestStopSignalFileCreation:
    """Tests for signal file creation with correct JSON payload."""

    def test_signal_file_created(self, cli_runner, temp_specs_dir):
        """stop writes a signal file with correct JSON payload."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "test-spec-001"],
        )
        assert result.exit_code == 0, f"Unexpected output: {result.output}"
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["data"]["spec_id"] == "test-spec-001"
        assert data["data"]["action"] == "stop_requested"

        # Verify signal file exists and has correct payload
        signal_file = temp_specs_dir / ".autonomy" / "signals" / "test-spec-001.stop"
        assert signal_file.exists()
        payload = json.loads(signal_file.read_text())
        assert payload["requested_by"] == "foundry-cli"
        assert payload["reason"] == "operator_stop"
        assert "requested_at" in payload

    def test_signal_file_message(self, cli_runner, temp_specs_dir):
        """stop emits human-readable message about signal file."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "signal_file" in data["data"]
        assert "test-spec-001.stop" in data["data"]["signal_file"]


class TestStopNoActiveSession:
    """Tests for error when spec not found."""

    def test_error_when_spec_not_found(self, cli_runner, temp_specs_dir):
        """stop returns error for non-existent spec."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "nonexistent-spec"],
        )
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["success"] is False
        assert data["data"]["error_code"] == "NOT_FOUND"


class TestStopForceMode:
    """Tests for --force mode process kill."""

    @patch("foundry_mcp.cli.commands.stop.subprocess.run")
    @patch("foundry_mcp.cli.commands.stop.os.kill")
    def test_force_kills_matching_processes(
        self, mock_kill, mock_run, cli_runner, temp_specs_dir
    ):
        """--force finds processes via pgrep and sends SIGTERM."""
        mock_run.return_value = SimpleNamespace(
            returncode=0, stdout="1234\n5678\n", stderr=""
        )

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "--force", "test-spec-001"],
        )
        assert result.exit_code == 0, f"Unexpected output: {result.output}"
        data = json.loads(result.output)
        assert data["data"]["action"] == "force_stop"
        assert 1234 in data["data"]["killed_pids"]
        assert 5678 in data["data"]["killed_pids"]
        assert mock_kill.call_count == 2

    @patch("foundry_mcp.cli.commands.stop.subprocess.run")
    def test_force_no_matching_processes(self, mock_run, cli_runner, temp_specs_dir):
        """--force with no matching processes returns empty killed list."""
        mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "--force", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["data"]["action"] == "force_stop"
        assert data["data"]["killed_pids"] == []


class TestStopForceRegexEscaping:
    """Tests for regex escaping in --force mode pgrep pattern."""

    @patch("foundry_mcp.cli.commands.stop.subprocess.run")
    def test_pgrep_pattern_escapes_regex_metacharacters(
        self, mock_run, cli_runner, temp_specs_dir
    ):
        """--force escapes regex metacharacters in spec_id for pgrep -f."""
        import re

        # Create a spec with regex metacharacters in the name
        active_dir = temp_specs_dir / "active"
        tricky_id = "spec.with+regex*chars"
        spec_data = {
            "id": tricky_id,
            "title": "Tricky",
            "version": "1.0.0",
            "status": "active",
            "hierarchy": {"spec-root": {"type": "root", "title": "T", "children": [], "status": "in_progress"}},
            "journal": [],
        }
        (active_dir / f"{tricky_id}.json").write_text(json.dumps(spec_data))

        # pgrep returns no matches
        mock_run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "--force", tricky_id],
        )
        assert result.exit_code == 0

        # Check that the pgrep call used re.escape(spec_id)
        pgrep_call = mock_run.call_args_list[0]
        pattern = pgrep_call[0][0][2]  # ["pgrep", "-f", <pattern>]
        assert re.escape(tricky_id) in pattern


class TestStopWaitMode:
    """Tests for --wait mode poll and exit on terminal status."""

    @patch("foundry_mcp.cli.commands.stop.time.sleep")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_wait_exits_on_terminal_status(
        self, mock_storage_cls, mock_sleep, cli_runner, temp_specs_dir
    ):
        """--wait polls and exits when session reaches terminal status."""
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        mock_storage.get_active_session.return_value = "session-123"
        mock_session = MagicMock()
        mock_session.status.value = "paused"
        mock_storage.load.return_value = mock_session

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "--wait", "test-spec-001"],
        )
        assert result.exit_code == 0, f"Unexpected output: {result.output}"
        data = json.loads(result.output)
        assert data["data"]["action"] == "stopped"
        assert data["data"]["final_status"] == "paused"

    @patch("foundry_mcp.cli.commands.stop.time.sleep")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_wait_exits_when_no_active_session(
        self, mock_storage_cls, mock_sleep, cli_runner, temp_specs_dir
    ):
        """--wait returns no_active_session when session is gone."""
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_active_session.return_value = None

        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "stop", "--wait", "test-spec-001"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["data"]["action"] == "stopped"
        assert data["data"]["final_status"] == "no_active_session"


class TestStopWaitTimeout:
    """Tests for --wait mode timeout."""

    @patch("foundry_mcp.cli.commands.stop.time.sleep")
    @patch("foundry_mcp.cli.commands.stop.time.monotonic")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_wait_times_out(
        self, mock_storage_cls, mock_monotonic, mock_sleep, cli_runner, temp_specs_dir
    ):
        """--wait returns timeout when deadline is exceeded."""
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        mock_storage.get_active_session.return_value = "session-123"
        mock_session = MagicMock()
        mock_session.status.value = "running"
        mock_storage.load.return_value = mock_session

        # Simulate time passing: first call sets deadline, second exceeds it
        mock_monotonic.side_effect = [0.0, 200.0]

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "stop",
                "--wait",
                "--timeout",
                "5",
                "test-spec-001",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["data"]["action"] == "timeout"
        assert data["data"]["final_status"] is None
        assert "Timed out" in data["data"]["message"]
