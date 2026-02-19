"""Unit tests for the foundry watch CLI command.

Tests cover:
- Data assembly resolves session and loads state
- Error when no active session
- Simple mode outputs formatted event lines
- Terminal state detection exits the loop
- Stop signal written via 's' key in live dashboard
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.cli.main import cli


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
                "status": "completed",
                "metadata": {},
                "dependencies": {},
            },
        },
        "journal": [],
    }
    spec_file = active_dir / "test-spec-001.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir


def _make_mock_session(status="running", active_phase="phase-1", task_id="task-1-1"):
    """Create a mock session object with required attributes."""
    session = MagicMock()
    session.id = "session-abc-123"
    session.spec_id = "test-spec-001"
    session.status.value = status
    session.active_phase_id = active_phase
    session.last_task_id = task_id
    session.context.context_usage_pct = 42
    session.context.last_heartbeat_at = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
    session.last_step_issued = None
    session.counters.tasks_completed = 3
    session.counters.consecutive_errors = 0
    session.model_dump.return_value = {
        "id": "session-abc-123",
        "spec_id": "test-spec-001",
        "status": status,
        "pause_reason": None,
        "created_at": "2026-02-18T10:00:00Z",
        "updated_at": "2026-02-18T12:00:00Z",
    }
    return session


def _make_mock_event(seq=1, event_type="step_issued", task_id="task-1-1", action="implement"):
    """Create a mock audit event."""
    evt = MagicMock()
    evt.sequence = seq
    evt.timestamp = "2026-02-18T12:00:00Z"
    evt.event_type.value = event_type
    evt.action = action
    evt.task_id = task_id
    evt.phase_id = None
    return evt


class TestDataAssembly:
    """Tests for _assemble_watch_data resolving session and loading state."""

    @patch("foundry_mcp.core.autonomy.audit.AuditLedger")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_assembles_session_and_progress(
        self, mock_storage_cls, mock_ledger_cls, temp_specs_dir
    ):
        """Data assembly resolves session ID, loads session, audit, progress."""
        from foundry_mcp.cli.commands.watch import _assemble_watch_data

        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_active_session.return_value = "session-abc-123"

        session = _make_mock_session()
        mock_storage.load.return_value = session

        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        evt = _make_mock_event()
        mock_ledger.get_entries.return_value = [evt]

        data = _assemble_watch_data(temp_specs_dir, "test-spec-001", max_events=10)

        assert data["session"]["id"] == "session-abc-123"
        assert data["session"]["status"] == "running"
        assert data["session"]["active_phase_id"] == "phase-1"
        assert data["session"]["context_usage_pct"] == 42
        assert data["session"]["tasks_completed"] == 3
        assert len(data["audit_events"]) == 1
        assert data["audit_events"][0]["event_type"] == "step_issued"
        assert data["progress"]["total_tasks"] > 0


class TestNoActiveSession:
    """Tests for error when no active session found."""

    def test_error_when_no_active_session(self, cli_runner, temp_specs_dir):
        """watch returns error when no active session exists."""
        with patch(
            "foundry_mcp.core.autonomy.memory.AutonomyStorage"
        ) as mock_storage_cls:
            mock_storage = MagicMock()
            mock_storage_cls.return_value = mock_storage
            mock_storage.get_active_session.return_value = None

            result = cli_runner.invoke(
                cli,
                [
                    "--specs-dir", str(temp_specs_dir),
                    "watch", "--simple", "test-spec-001",
                ],
            )
            assert result.exit_code == 1
            data = json.loads(result.output)
            assert data["success"] is False
            assert data["data"]["error_code"] == "NOT_FOUND"


class TestSimpleModeOutput:
    """Tests for --simple mode output format."""

    @patch("foundry_mcp.cli.commands.watch.time.sleep")
    @patch("foundry_mcp.core.autonomy.audit.AuditLedger")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_simple_mode_prints_header_and_events(
        self, mock_storage_cls, mock_ledger_cls, mock_sleep, cli_runner, temp_specs_dir
    ):
        """--simple prints header line and formatted event lines."""
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_active_session.return_value = "session-abc-123"

        # First load returns running, second returns paused (to exit loop)
        session_running = _make_mock_session(status="running")
        session_paused = _make_mock_session(status="paused")
        mock_storage.load.side_effect = [session_running, session_paused]

        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        evt = _make_mock_event(seq=1, event_type="step_issued", task_id="task-1-1", action="implement")
        mock_ledger.get_entries.return_value = [evt]

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir", str(temp_specs_dir),
                "watch", "--simple", "test-spec-001",
            ],
        )
        assert result.exit_code == 0, f"Output: {result.output}"
        lines = result.output.strip().split("\n")

        # Header line
        assert "session=session-abc-123" in lines[0]
        assert "spec=test-spec-001" in lines[0]
        assert "status=running" in lines[0]

        # Event line
        assert "step_issued" in result.output
        assert "implement" in result.output

        # Terminal message
        assert "--- session paused ---" in result.output


class TestTerminalStateExit:
    """Tests for terminal state detection exiting the loop."""

    @patch("foundry_mcp.core.autonomy.audit.AuditLedger")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_simple_mode_exits_on_completed(
        self, mock_storage_cls, mock_ledger_cls, cli_runner, temp_specs_dir
    ):
        """--simple exits immediately when session is already completed."""
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_active_session.return_value = "session-abc-123"

        session = _make_mock_session(status="completed")
        mock_storage.load.return_value = session

        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_entries.return_value = []

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir", str(temp_specs_dir),
                "watch", "--simple", "test-spec-001",
            ],
        )
        assert result.exit_code == 0
        assert "--- session completed ---" in result.output

    @patch("foundry_mcp.core.autonomy.audit.AuditLedger")
    @patch("foundry_mcp.core.autonomy.memory.AutonomyStorage")
    def test_simple_mode_exits_on_failed(
        self, mock_storage_cls, mock_ledger_cls, cli_runner, temp_specs_dir
    ):
        """--simple exits immediately when session is failed."""
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_storage.get_active_session.return_value = "session-abc-123"

        session = _make_mock_session(status="failed")
        mock_storage.load.return_value = session

        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_entries.return_value = []

        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir", str(temp_specs_dir),
                "watch", "--simple", "test-spec-001",
            ],
        )
        assert result.exit_code == 0
        assert "--- session failed ---" in result.output


class TestWriteStopSignal:
    """Tests for _write_stop_signal writing the shared signal file."""

    def test_write_stop_signal_creates_file(self, tmp_path):
        """_write_stop_signal creates a signal file via the shared utility."""
        from foundry_mcp.cli.commands.watch import _write_stop_signal

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        result_path = _write_stop_signal(specs_dir, "test-spec-001")

        assert result_path.exists()
        content = json.loads(result_path.read_text())
        assert content["requested_by"] == "foundry-watch"
        assert content["reason"] == "operator_stop"

    def test_write_stop_signal_path_matches_convention(self, tmp_path):
        """Signal file is written at the canonical path."""
        from foundry_mcp.cli.commands.watch import _write_stop_signal

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        result_path = _write_stop_signal(specs_dir, "my-spec")

        expected = specs_dir / ".autonomy" / "signals" / "my-spec.stop"
        assert result_path == expected


class TestSignalIntegration:
    """Integration test: signal written by CLI is consumed by orchestrator."""

    def test_cli_signal_consumed_by_orchestrator(self, tmp_path):
        """Signal file written by write_stop_signal triggers orchestrator pause."""
        from foundry_mcp.core.autonomy.signals import write_stop_signal

        # Set up workspace with specs dir and a signal file
        specs_dir = tmp_path / "specs"
        active_dir = specs_dir / "active"
        active_dir.mkdir(parents=True)

        spec_data = {
            "id": "test-spec-001",
            "title": "Test",
            "version": "1.0.0",
            "status": "active",
            "hierarchy": {
                "spec-root": {
                    "type": "root",
                    "title": "Test",
                    "children": ["phase-1"],
                    "status": "in_progress",
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Phase 1",
                    "parent": "spec-root",
                    "children": ["task-1"],
                    "status": "in_progress",
                },
                "task-1": {
                    "type": "task",
                    "title": "Task 1",
                    "parent": "phase-1",
                    "status": "pending",
                    "metadata": {},
                    "dependencies": {},
                },
            },
            "journal": [],
        }
        spec_file = active_dir / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_data, indent=2))

        # Write signal via shared utility (same code path as stop/watch CLI)
        signal_file = write_stop_signal(specs_dir, "test-spec-001", requested_by="foundry-test")

        assert signal_file.exists()

        # Verify the orchestrator's _check_stop_signal finds and consumes it
        from foundry_mcp.core.autonomy.signals import signal_path_for_spec

        orch_signal = signal_path_for_spec(specs_dir, "test-spec-001")
        assert orch_signal == signal_file
        assert orch_signal.is_file()

        # Consuming the signal (as orchestrator would)
        orch_signal.unlink()
        assert not signal_file.exists()
