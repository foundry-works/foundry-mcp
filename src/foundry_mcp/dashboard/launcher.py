"""Dashboard launcher with subprocess management.

Manages the Streamlit server as a subprocess, allowing the dashboard to be
started and stopped from the CLI or programmatically.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Global process reference
_dashboard_process: Optional[subprocess.Popen] = None


def launch_dashboard(
    host: str = "127.0.0.1",
    port: int = 8501,
    open_browser: bool = True,
) -> dict:
    """Launch the Streamlit dashboard server.

    Args:
        host: Host to bind to (default: localhost only)
        port: Port to run on (default: 8501)
        open_browser: Whether to open browser automatically

    Returns:
        dict with:
            - success: bool
            - url: Dashboard URL
            - pid: Process ID
            - message: Status message
    """
    global _dashboard_process

    # Check if already running
    status = get_dashboard_status()
    if status.get("running"):
        return {
            "success": True,
            "url": f"http://{host}:{port}",
            "pid": status["pid"],
            "message": "Dashboard already running",
        }

    # Path to the main Streamlit app
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        return {
            "success": False,
            "message": f"Dashboard app not found at {app_path}",
        }

    # Build Streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "dark",
    ]

    if not open_browser:
        cmd.extend(["--server.headless", "true"])

    # Environment with dashboard mode flag
    env = {
        **os.environ,
        "FOUNDRY_MCP_DASHBOARD_MODE": "1",
    }

    try:
        # Start subprocess
        _dashboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Brief wait to check for immediate failures
        time.sleep(1)
        poll = _dashboard_process.poll()
        if poll is not None:
            stderr = _dashboard_process.stderr.read().decode() if _dashboard_process.stderr else ""
            return {
                "success": False,
                "message": f"Dashboard failed to start (exit code {poll}): {stderr}",
            }

        logger.info("Dashboard started at http://%s:%s (pid=%s)", host, port, _dashboard_process.pid)

        return {
            "success": True,
            "url": f"http://{host}:{port}",
            "pid": _dashboard_process.pid,
            "message": "Dashboard started successfully",
        }

    except FileNotFoundError:
        return {
            "success": False,
            "message": "Streamlit not installed. Install with: pip install foundry-mcp[dashboard]",
        }
    except Exception as e:
        logger.exception("Failed to start dashboard")
        return {
            "success": False,
            "message": f"Failed to start dashboard: {e}",
        }


def stop_dashboard() -> dict:
    """Stop the running dashboard server.

    Returns:
        dict with:
            - success: bool
            - message: Status message
    """
    global _dashboard_process

    if _dashboard_process is None:
        return {
            "success": False,
            "message": "No dashboard process to stop",
        }

    try:
        _dashboard_process.terminate()
        _dashboard_process.wait(timeout=5)
        pid = _dashboard_process.pid
        _dashboard_process = None

        logger.info("Dashboard stopped (pid=%s)", pid)

        return {
            "success": True,
            "message": f"Dashboard stopped (pid={pid})",
        }

    except subprocess.TimeoutExpired:
        _dashboard_process.kill()
        _dashboard_process = None
        return {
            "success": True,
            "message": "Dashboard killed (did not terminate gracefully)",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to stop dashboard: {e}",
        }


def get_dashboard_status() -> dict:
    """Check if dashboard is running.

    Returns:
        dict with:
            - running: bool
            - pid: Process ID (if running)
            - exit_code: Exit code (if not running)
    """
    global _dashboard_process

    if _dashboard_process is None:
        return {"running": False}

    poll = _dashboard_process.poll()
    if poll is not None:
        return {"running": False, "exit_code": poll}

    return {"running": True, "pid": _dashboard_process.pid}
