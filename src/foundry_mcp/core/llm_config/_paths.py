"""Config file search path helpers shared across all LLM config domains."""

import os
from pathlib import Path
from typing import List, Optional


def _get_bundled_default_config() -> Optional[Path]:
    """Locate the bundled default config shipped with the package.

    Returns:
        Path to the bundled default-config.toml, or None if not found.
    """
    # In development: samples/foundry-mcp.toml relative to repo root
    # In installed package: foundry_mcp/resources/default-config.toml
    resources_dir = Path(__file__).parent.parent.parent / "resources"
    bundled = resources_dir / "default-config.toml"
    if bundled.exists():
        return bundled

    # Development fallback: walk up from this file to find samples/
    candidate = Path(__file__).parent.parent.parent.parent.parent / "samples" / "foundry-mcp.toml"
    if candidate.exists():
        return candidate

    return None


def _default_config_search_paths() -> List[Path]:
    """Return the standard config file search paths (lowest to highest priority).

    Order matches ServerConfig.load() for consistency:
    1. Bundled sample defaults (last resort)
    2. XDG config (~/.config/foundry-mcp/config.toml)
    3. User home config (~/.foundry-mcp.toml)
    4. Project dotfile config (./.foundry-mcp.toml)
    5. Project config (./foundry-mcp.toml)
    """
    paths: List[Path] = []

    # Bundled sample defaults (last resort)
    bundled = _get_bundled_default_config()
    if bundled:
        paths.append(bundled)

    # XDG config
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    paths.append(Path(xdg_config_home) / "foundry-mcp" / "config.toml")

    # User home config
    paths.append(Path.home() / ".foundry-mcp.toml")

    # Project configs (cwd)
    paths.append(Path(".foundry-mcp.toml"))
    paths.append(Path("foundry-mcp.toml"))

    return paths
