"""Configuration management for foundry-mcp-ctl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

Mode = Literal["minimal", "full"]

DEFAULT_MODE: Mode = "minimal"
CONFIG_DIR = Path.home() / ".foundry-mcp-ctl"


def get_config_path() -> Path:
    """Get path to the config file."""
    return CONFIG_DIR / "config.json"


def get_restart_signal_path(server_name: str) -> Path:
    """Get path to the restart signal file for a server."""
    return CONFIG_DIR / f"{server_name}.restart"


def ensure_config_dir() -> None:
    """Ensure the config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_mode() -> Mode:
    """Read current mode from config file."""
    config_path = get_config_path()
    if not config_path.exists():
        return DEFAULT_MODE

    try:
        with open(config_path) as f:
            data = json.load(f)
        mode = data.get("mode", DEFAULT_MODE)
        return mode if mode in ("minimal", "full") else DEFAULT_MODE
    except (json.JSONDecodeError, IOError):
        return DEFAULT_MODE


def set_mode(mode: Mode) -> None:
    """Write mode to config file."""
    ensure_config_dir()
    config_path = get_config_path()

    data = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    data["mode"] = mode

    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)


def signal_restart(server_name: str) -> None:
    """Write restart signal file."""
    ensure_config_dir()
    signal_path = get_restart_signal_path(server_name)
    signal_path.write_text(get_mode())


def clear_restart_signal(server_name: str) -> None:
    """Remove restart signal file."""
    signal_path = get_restart_signal_path(server_name)
    if signal_path.exists():
        signal_path.unlink()
