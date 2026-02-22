"""ServerConfig dataclass and global configuration state.

This module defines the ``ServerConfig`` class (field declarations and simple
accessor methods) and the global ``get_config`` / ``set_config`` helpers.
Loading and validation logic lives in the ``_ServerConfigLoader`` mixin
(``loader.py``) which ``ServerConfig`` inherits from.
"""

import logging
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import List, Optional, TypeVar

from foundry_mcp.config.autonomy import (
    AutonomyPostureConfig,
    AutonomySecurityConfig,
    AutonomySessionDefaultsConfig,
)
from foundry_mcp.config.domains import (
    ErrorCollectionConfig,
    GitSettings,
    HealthConfig,
    MetricsPersistenceConfig,
    ObservabilityConfig,
    TestConfig,
)
from foundry_mcp.config.loader import _ServerConfigLoader
from foundry_mcp.config.research import ResearchConfig


def _get_version() -> str:
    """Get package version from metadata (single source of truth: pyproject.toml)."""
    try:
        return get_package_version("foundry-mcp")
    except PackageNotFoundError:
        return "0.5.0"  # Fallback for dev without install


_PACKAGE_VERSION = _get_version()

T = TypeVar("T")


@dataclass
class ServerConfig(_ServerConfigLoader):
    """Server configuration with support for env vars and TOML overrides."""

    # Workspace configuration
    workspace_roots: List[Path] = field(default_factory=list)
    specs_dir: Optional[Path] = None
    research_dir: Optional[Path] = None  # Research state storage (default: specs/.research)

    # Logging configuration
    log_level: str = "INFO"
    structured_logging: bool = True

    # Authentication configuration
    api_keys: List[str] = field(default_factory=list)
    require_auth: bool = False

    # Server configuration
    server_name: str = "foundry-mcp"
    server_version: str = field(default_factory=lambda: _PACKAGE_VERSION)

    # Git workflow configuration
    git: GitSettings = field(default_factory=GitSettings)

    # Observability configuration
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Health check configuration
    health: HealthConfig = field(default_factory=HealthConfig)

    # Error collection configuration
    error_collection: ErrorCollectionConfig = field(default_factory=ErrorCollectionConfig)

    # Metrics persistence configuration
    metrics_persistence: MetricsPersistenceConfig = field(default_factory=MetricsPersistenceConfig)

    # Test runner configuration
    test: TestConfig = field(default_factory=TestConfig)

    # Research workflows configuration
    research: ResearchConfig = field(default_factory=ResearchConfig)

    # Autonomy security configuration
    autonomy_security: AutonomySecurityConfig = field(default_factory=AutonomySecurityConfig)
    autonomy_posture: AutonomyPostureConfig = field(default_factory=AutonomyPostureConfig)
    autonomy_session_defaults: AutonomySessionDefaultsConfig = field(default_factory=AutonomySessionDefaultsConfig)

    # Tool registration control
    disabled_tools: List[str] = field(default_factory=list)

    startup_warnings: List[str] = field(default_factory=list, repr=False)

    def _add_startup_warning(self, message: str) -> None:
        if message and message not in self.startup_warnings:
            self.startup_warnings.append(message)

    def validate_api_key(self, key: Optional[str]) -> bool:
        """
        Validate an API key.

        Args:
            key: API key to validate

        Returns:
            True if valid (or auth not required), False otherwise
        """
        if not self.require_auth:
            return True

        if not key:
            return False

        return key in self.api_keys

    def get_research_dir(self, specs_dir: Optional[Path] = None) -> Path:
        """
        Get the resolved research directory path.

        Priority:
        1. Explicitly configured research_dir (from TOML or env var)
        2. Default: specs_dir/.research (where specs_dir is resolved)

        Args:
            specs_dir: Optional specs directory to use for default path.
                      If not provided, uses self.specs_dir or "./specs"

        Returns:
            Path to research directory
        """
        if self.research_dir is not None:
            return self.research_dir.expanduser()

        # Fall back to default: specs/.research
        base_specs = specs_dir or self.specs_dir or Path("./specs")
        return base_specs / ".research"

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        level = getattr(logging, self.log_level, logging.INFO)

        if self.structured_logging:
            # JSON-style structured logging
            formatter = logging.Formatter(
                '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
            )
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        root_logger = logging.getLogger("foundry_mcp")
        root_logger.setLevel(level)
        root_logger.addHandler(handler)


# Global configuration instance
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ServerConfig.from_env()
    return _config


def set_config(config: ServerConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
