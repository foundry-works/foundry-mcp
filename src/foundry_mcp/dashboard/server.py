"""
Dashboard HTTP server for foundry-mcp.

Provides a lightweight web server for the observability dashboard,
serving static files and API endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Any, Optional

from foundry_mcp.config import DashboardConfig

logger = logging.getLogger(__name__)

# Global server state
_server_thread: Optional[threading.Thread] = None
_external_dashboard: bool = False  # True if dashboard is running externally
_server_loop: Optional[asyncio.AbstractEventLoop] = None
_server_running = threading.Event()
_current_config: Optional[DashboardConfig] = None


def _check_existing_dashboard(host: str, port: int) -> bool:
    """Check if a foundry-mcp dashboard is already running at the given address.

    Makes a quick HTTP request to the health endpoint to verify it's our dashboard.

    Args:
        host: Host to check
        port: Port to check

    Returns:
        True if a foundry-mcp dashboard is already running, False otherwise
    """
    import socket
    import urllib.request
    import urllib.error
    import json

    # First check if port is in use at all
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        result = sock.connect_ex((host, port))
        if result != 0:
            # Port not in use
            return False
    finally:
        sock.close()

    # Port is in use - check if it's our dashboard by looking for our health endpoint
    try:
        url = f"http://{host}:{port}/api/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as response:
            data = response.read().decode("utf-8")
            # Check if response is JSON with expected health structure
            try:
                health = json.loads(data)
                # Our health endpoint returns status, is_healthy, dependencies
                if "status" in health and "is_healthy" in health:
                    return True
            except json.JSONDecodeError:
                pass
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        pass

    return False


def get_static_dir() -> Path:
    """Get the path to static files directory."""
    return Path(__file__).parent / "static"


def get_dashboard_url(config: Optional[DashboardConfig] = None) -> str:
    """Get the dashboard URL.

    Args:
        config: Optional config (uses global if not provided)

    Returns:
        Dashboard URL string
    """
    cfg = config or _current_config
    if cfg is None:
        return "http://127.0.0.1:8080"
    return f"http://{cfg.host}:{cfg.port}"


async def _create_app(config: DashboardConfig) -> Any:
    """Create the aiohttp application.

    Args:
        config: Dashboard configuration

    Returns:
        aiohttp Application instance
    """
    try:
        from aiohttp import web
    except ImportError:
        raise ImportError(
            "aiohttp is required for the dashboard. "
            "Install with: pip install foundry-mcp[dashboard]"
        )

    from foundry_mcp.dashboard.api import setup_api_routes

    app = web.Application()

    # Store config in app for access by handlers
    app["config"] = config

    # Setup API routes
    setup_api_routes(app)

    # Setup static file serving
    static_dir = get_static_dir()
    if static_dir.exists():
        app.router.add_static("/static", static_dir, name="static")

        # Serve index.html at root
        async def index_handler(request: web.Request) -> web.Response:
            index_path = static_dir / "index.html"
            if index_path.exists():
                return web.FileResponse(index_path)
            return web.Response(
                text="Dashboard not found. Static files may not be installed.",
                status=404,
            )

        app.router.add_get("/", index_handler)
    else:
        logger.warning(f"Static directory not found: {static_dir}")

        async def placeholder_handler(request: web.Request) -> web.Response:
            return web.Response(
                text="<html><body><h1>foundry-mcp Dashboard</h1>"
                "<p>Static files not installed.</p></body></html>",
                content_type="text/html",
            )

        app.router.add_get("/", placeholder_handler)

    return app


async def _run_server(config: DashboardConfig) -> None:
    """Run the dashboard server.

    Args:
        config: Dashboard configuration
    """
    try:
        from aiohttp import web
    except ImportError:
        logger.error("aiohttp not installed. Dashboard unavailable.")
        return

    app = await _create_app(config)
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)

    try:
        await site.start()
        logger.info(f"Dashboard started at http://{config.host}:{config.port}")
        _server_running.set()

        # Keep running until stopped
        while _server_running.is_set():
            await asyncio.sleep(0.5)

    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {config.port} is already in use. Dashboard not started.")
        else:
            logger.error(f"Failed to start dashboard: {e}")
    finally:
        await runner.cleanup()


def _server_thread_main(config: DashboardConfig) -> None:
    """Main function for the server thread.

    Args:
        config: Dashboard configuration
    """
    global _server_loop

    _server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_server_loop)

    try:
        _server_loop.run_until_complete(_run_server(config))
    except Exception as e:
        logger.error(f"Dashboard server error: {e}")
    finally:
        _server_loop.close()
        _server_loop = None


def start_dashboard(config: DashboardConfig) -> bool:
    """Start the dashboard server in a background thread.

    If a dashboard is already running on the configured port (from a previous
    server instance), this function will detect it and skip startup rather
    than fail with a port binding error.

    Args:
        config: Dashboard configuration

    Returns:
        True if started successfully or already running, False otherwise
    """
    global _server_thread, _current_config, _external_dashboard

    if not config.enabled:
        logger.debug("Dashboard is disabled in configuration")
        return False

    if _server_thread is not None and _server_thread.is_alive():
        logger.debug("Dashboard server is already running in this process")
        return True

    if _external_dashboard:
        logger.debug("Using existing external dashboard")
        return True

    # Check if a dashboard is already running on the port (e.g., from previous server)
    if _check_existing_dashboard(config.host, config.port):
        url = f"http://{config.host}:{config.port}"
        logger.info(f"Found existing dashboard at {url}, skipping startup")
        _external_dashboard = True
        _current_config = config
        return True

    try:
        # Check if aiohttp is available
        import aiohttp  # noqa: F401
    except ImportError:
        logger.warning(
            "aiohttp not installed. Dashboard unavailable. "
            "Install with: pip install foundry-mcp[dashboard]"
        )
        return False

    _current_config = config
    _server_running.clear()

    _server_thread = threading.Thread(
        target=_server_thread_main,
        args=(config,),
        name="dashboard-server",
        daemon=True,
    )
    _server_thread.start()

    # Wait for server to start (with short timeout - don't block MCP startup)
    if _server_running.wait(timeout=2.0):
        url = get_dashboard_url(config)
        logger.info(f"Dashboard available at {url}")

        if config.auto_open_browser:
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Failed to open browser: {e}")

        return True
    else:
        logger.error("Dashboard server failed to start within timeout")
        return False


def stop_dashboard() -> None:
    """Stop the dashboard server."""
    global _server_thread, _current_config, _external_dashboard

    # Don't try to stop external dashboards
    if _external_dashboard:
        logger.debug("External dashboard - not stopping")
        _external_dashboard = False
        _current_config = None
        return

    if _server_thread is None:
        return

    _server_running.clear()

    # Wait for thread to finish
    if _server_thread.is_alive():
        _server_thread.join(timeout=5.0)
        if _server_thread.is_alive():
            logger.warning("Dashboard server thread did not stop cleanly")

    _server_thread = None
    _current_config = None
    logger.info("Dashboard server stopped")


def is_dashboard_running() -> bool:
    """Check if the dashboard server is running.

    Returns:
        True if running (either locally or externally), False otherwise
    """
    return _server_running.is_set() or _external_dashboard
