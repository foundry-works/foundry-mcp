"""Shared signal file utilities for autonomy stop/pause coordination.

The stop signal protocol uses a well-known file path to communicate
between the CLI (writer) and the orchestrator (consumer):

    specs/.autonomy/signals/{spec_id}.stop

This module provides helpers for both sides of the protocol.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# Session statuses that indicate the session has reached a terminal/stopped state.
# Shared between stop, watch, and any other CLI command that polls session state.
TERMINAL_STATUSES: frozenset[str] = frozenset({"paused", "completed", "ended", "failed"})


def signal_dir_for_specs(specs_dir: Path) -> Path:
    """Return the canonical signal directory for a given specs dir."""
    return specs_dir / ".autonomy" / "signals"


def signal_path_for_spec(specs_dir: Path, spec_id: str) -> Path:
    """Return the canonical stop-signal file path for a spec."""
    return signal_dir_for_specs(specs_dir) / f"{spec_id}.stop"


def write_stop_signal(
    specs_dir: Path, spec_id: str, requested_by: str = "foundry-cli"
) -> Path:
    """Write a stop signal file for the given spec.

    Args:
        specs_dir: Path to the specs directory.
        spec_id: Spec identifier.
        requested_by: Identifier of the component requesting the stop.

    Returns:
        Path to the written signal file.
    """
    sig_dir = signal_dir_for_specs(specs_dir)
    sig_dir.mkdir(parents=True, exist_ok=True)

    sig_file = sig_dir / f"{spec_id}.stop"
    payload = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": requested_by,
        "reason": "operator_stop",
    }
    sig_file.write_text(json.dumps(payload, indent=2))
    return sig_file
