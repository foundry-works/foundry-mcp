"""
Standalone dashboard entry point.

Run with: python -m foundry_mcp.dashboard

The dashboard is decoupled from the MCP server and can run independently.
It reads from the same metrics and error stores as the MCP server.
"""

import argparse
import logging
import signal
import sys

from foundry_mcp.config import get_config
from foundry_mcp.dashboard.server import start_dashboard, stop_dashboard, get_dashboard_url


def main() -> None:
    """Run the dashboard as a standalone server."""
    parser = argparse.ArgumentParser(
        description="Foundry MCP Dashboard - Standalone observability UI"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to run dashboard on (default: from config or 8080)"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load config
    config = get_config()
    dashboard_config = config.dashboard

    # Override with CLI args
    if args.port:
        dashboard_config.port = args.port
    if args.host:
        dashboard_config.host = args.host
    if args.no_browser:
        dashboard_config.auto_open_browser = False

    # Force enable (user is explicitly running dashboard)
    dashboard_config.enabled = True

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        stop_dashboard()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start dashboard
    logger.info("Starting Foundry MCP Dashboard...")
    success = start_dashboard(dashboard_config)

    if not success:
        logger.error("Failed to start dashboard")
        sys.exit(1)

    url = get_dashboard_url(dashboard_config)
    logger.info(f"Dashboard running at {url}")
    logger.info("Press Ctrl+C to stop")

    # Keep running until interrupted
    # Note: signal.pause() is unreliable - it returns on ANY signal,
    # including SIGCHLD from webbrowser.open() spawning a subprocess.
    # Use a simple sleep loop instead for cross-platform reliability.
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
