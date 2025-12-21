"""CLI entry point for foundry-mcp-ctl."""

from __future__ import annotations

import sys
from typing import List

import click

from . import __version__
from .config import clear_restart_signal, get_mode, set_mode, signal_restart
from .helper import run_helper
from .wrapper import run_wrapper


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Foundry MCP Control - programmatic mode toggling."""
    pass


@main.command()
@click.option("--name", required=True, help="Server name for restart signaling")
@click.argument("command", nargs=-1, required=True)
def wrap(name: str, command: tuple) -> None:
    """Wrap foundry-mcp and monitor for restart signals.

    Example:
        foundry-mcp-ctl wrap --name foundry-mcp -- python -m foundry_mcp.server
    """
    sys.exit(run_wrapper(name, list(command)))


@main.command()
def helper() -> None:
    """Run the helper MCP server with set_mode tool."""
    run_helper()


@main.command("set-mode")
@click.argument("mode", type=click.Choice(["minimal", "full"]))
@click.option("--server", default="foundry-mcp", help="Server name to restart")
def set_mode_cmd(mode: str, server: str) -> None:
    """Set mode and signal restart (for CLI use)."""
    set_mode(mode)  # type: ignore
    signal_restart(server)
    click.echo(f"Mode set to {mode}, restart signaled for {server}")


@main.command()
@click.option("--server", default="foundry-mcp", help="Server name")
def restart(server: str) -> None:
    """Signal server restart without changing mode."""
    signal_restart(server)
    click.echo(f"Restart signaled for {server}")


@main.command()
@click.option("--server", default="foundry-mcp", help="Server name")
def clean(server: str) -> None:
    """Clear restart signal file."""
    clear_restart_signal(server)
    click.echo(f"Restart signal cleared for {server}")


@main.command()
def status() -> None:
    """Show current mode."""
    mode = get_mode()
    tools = 17 if mode == "full" else 1
    click.echo(f"Mode: {mode}")
    click.echo(f"Tools available: {tools}")


if __name__ == "__main__":
    main()
