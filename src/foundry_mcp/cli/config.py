"""CLI configuration and workspace detection.

Provides configuration handling for the SDD CLI, leveraging the
shared foundry_mcp.config module and core spec utilities.
"""

from pathlib import Path
from typing import Optional

from foundry_mcp.config import ServerConfig, get_config as get_server_config
from foundry_mcp.core.spec import find_specs_directory


class CLIContext:
    """CLI execution context with resolved configuration.

    Holds the effective configuration for a CLI command, including
    any overrides from command-line options.
    """

    def __init__(
        self,
        specs_dir: Optional[str] = None,
        server_config: Optional[ServerConfig] = None,
    ):
        """Initialize CLI context.

        Args:
            specs_dir: Explicit specs directory override from --specs-dir.
            server_config: Optional server config (uses global if not provided).
        """
        self._specs_dir_override = specs_dir
        self._config = server_config or get_server_config()
        self._resolved_specs_dir: Optional[Path] = None

    @property
    def specs_dir(self) -> Optional[Path]:
        """Get the resolved specs directory.

        Resolution order:
        1. CLI --specs-dir option (highest priority)
        2. ServerConfig.specs_dir (from env/TOML)
        3. Auto-detected via find_specs_directory()
        """
        if self._resolved_specs_dir is not None:
            return self._resolved_specs_dir

        # CLI override takes priority
        if self._specs_dir_override:
            self._resolved_specs_dir = Path(self._specs_dir_override).resolve()
            return self._resolved_specs_dir

        # Server config next
        if self._config.specs_dir:
            self._resolved_specs_dir = self._config.specs_dir.resolve()
            return self._resolved_specs_dir

        # Auto-detect
        detected = find_specs_directory()
        if detected:
            self._resolved_specs_dir = detected
            return self._resolved_specs_dir

        return None

    @property
    def config(self) -> ServerConfig:
        """Get the underlying server configuration."""
        return self._config

    def require_specs_dir(self) -> Path:
        """Get specs directory, raising if not found.

        Returns:
            Resolved specs directory path.

        Raises:
            FileNotFoundError: If no specs directory could be resolved.
        """
        specs = self.specs_dir
        if specs is None:
            raise FileNotFoundError(
                "No specs directory found. "
                "Use --specs-dir or set SDD_SPECS_DIR environment variable."
            )
        return specs


def create_context(specs_dir: Optional[str] = None) -> CLIContext:
    """Create a CLI context with optional overrides.

    Args:
        specs_dir: Optional specs directory override.

    Returns:
        Configured CLIContext instance.
    """
    return CLIContext(specs_dir=specs_dir)
