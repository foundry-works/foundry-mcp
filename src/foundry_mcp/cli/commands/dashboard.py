"""Dashboard CLI commands.

Provides commands for starting, stopping, and managing the Streamlit dashboard.
"""

import click

from foundry_mcp.cli.output import emit, emit_error


@click.group("dashboard")
def dashboard_group() -> None:
    """Dashboard server management commands.

    Start, stop, and manage the Streamlit-based observability dashboard.
    """
    pass


@dashboard_group.command("start")
@click.option(
    "--port",
    "-p",
    default=8501,
    type=int,
    help="Port to run dashboard on (default: 8501)",
)
@click.option(
    "--host",
    "-H",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1 for localhost only)",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't automatically open browser",
)
@click.pass_context
def dashboard_start_cmd(
    ctx: click.Context,
    port: int,
    host: str,
    no_browser: bool,
) -> None:
    """Start the Streamlit dashboard server.

    Launches the dashboard in a background process and optionally opens
    your browser to view it.

    Examples:
        # Start with defaults (localhost:8501, opens browser)
        foundry-cli dashboard start

        # Start on custom port without browser
        foundry-cli dashboard start --port 8080 --no-browser

        # Expose to network (be careful with security)
        foundry-cli dashboard start --host 0.0.0.0
    """
    try:
        from foundry_mcp.dashboard import launch_dashboard

        result = launch_dashboard(
            host=host,
            port=port,
            open_browser=not no_browser,
        )

        if result.get("success"):
            emit(
                {
                    "success": True,
                    "message": result.get("message", "Dashboard started"),
                    "url": result.get("url"),
                    "pid": result.get("pid"),
                }
            )
        else:
            emit_error(result.get("message", "Failed to start dashboard"))

    except ImportError:
        emit_error(
            "Dashboard dependencies not installed. "
            "Install with: pip install foundry-mcp[dashboard]"
        )
    except Exception as e:
        emit_error(f"Failed to start dashboard: {e}")


@dashboard_group.command("stop")
@click.pass_context
def dashboard_stop_cmd(ctx: click.Context) -> None:
    """Stop the running dashboard server.

    Terminates the dashboard process if one is running.
    """
    try:
        from foundry_mcp.dashboard import stop_dashboard

        result = stop_dashboard()

        if result.get("success"):
            emit(
                {
                    "success": True,
                    "message": result.get("message", "Dashboard stopped"),
                }
            )
        else:
            emit(
                {
                    "success": False,
                    "message": result.get("message", "No dashboard to stop"),
                }
            )

    except ImportError:
        emit_error("Dashboard module not available")
    except Exception as e:
        emit_error(f"Failed to stop dashboard: {e}")


@dashboard_group.command("status")
@click.pass_context
def dashboard_status_cmd(ctx: click.Context) -> None:
    """Check if dashboard is running.

    Shows the current status of the dashboard server process.
    """
    try:
        from foundry_mcp.dashboard import get_dashboard_status

        status = get_dashboard_status()

        emit(
            {
                "running": status.get("running", False),
                "pid": status.get("pid"),
                "exit_code": status.get("exit_code"),
            }
        )

    except ImportError:
        emit({"running": False, "message": "Dashboard module not available"})
    except Exception as e:
        emit_error(f"Failed to get dashboard status: {e}")
