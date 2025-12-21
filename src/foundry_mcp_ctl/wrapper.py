"""Wrapper that launches foundry-mcp and monitors for restart signals."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import (
    clear_restart_signal,
    get_mode,
    get_restart_signal_path,
)


def clear_pycache(base_path: Path) -> None:
    """Clear __pycache__ directories under base_path."""
    for pycache in base_path.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def graceful_shutdown(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Shutdown subprocess gracefully, escalating to SIGKILL if needed."""
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class Wrapper:
    """Wrapper that manages foundry-mcp subprocess lifecycle."""

    def __init__(self, server_name: str, command: List[str]):
        self.server_name = server_name
        self.command = command
        self.process: Optional[subprocess.Popen] = None
        self.running = True

    def start_process(self) -> subprocess.Popen:
        """Start the foundry-mcp subprocess with current mode."""
        env = os.environ.copy()
        env["FOUNDRY_MODE"] = get_mode()

        # Clear pycache to ensure fresh module loading
        foundry_mcp_path = Path(__file__).parent.parent / "foundry_mcp"
        if foundry_mcp_path.exists():
            clear_pycache(foundry_mcp_path)

        return subprocess.Popen(
            self.command,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )

    def check_restart_signal(self) -> bool:
        """Check if restart signal file exists."""
        signal_path = get_restart_signal_path(self.server_name)
        return signal_path.exists()

    def handle_restart(self) -> None:
        """Handle restart by stopping and starting the subprocess."""
        clear_restart_signal(self.server_name)

        if self.process:
            graceful_shutdown(self.process)

        self.process = self.start_process()

    def run(self) -> int:
        """Run the wrapper, managing the subprocess lifecycle."""
        # Setup signal handlers
        def handle_signal(signum: int, frame) -> None:
            self.running = False
            if self.process:
                graceful_shutdown(self.process)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Start initial process
        self.process = self.start_process()

        # Monitor loop
        while self.running:
            # Check if process has exited
            if self.process.poll() is not None:
                # Process exited, check if it was a restart
                if self.check_restart_signal():
                    self.handle_restart()
                else:
                    # Normal exit
                    return self.process.returncode

            # Check for restart signal
            if self.check_restart_signal():
                self.handle_restart()

            time.sleep(0.1)

        return 0


def run_wrapper(server_name: str, command: List[str]) -> int:
    """Run the wrapper with given server name and command."""
    wrapper = Wrapper(server_name, command)
    return wrapper.run()
