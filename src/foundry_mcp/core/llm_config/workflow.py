"""Workflow configuration: WorkflowMode, WorkflowConfig and load/get/set/reset functions."""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

logger = logging.getLogger(__name__)


class WorkflowMode(str, Enum):
    """Workflow execution modes.

    SINGLE: Execute one task at a time with user approval between tasks
    AUTONOMOUS: Execute all tasks in phase automatically until completion or blocker
    BATCH: Execute a specified number of tasks, then pause for review
    """

    SINGLE = "single"
    AUTONOMOUS = "autonomous"
    BATCH = "batch"


@dataclass
class WorkflowConfig:
    """Workflow configuration parsed from foundry-mcp.toml [workflow] section.

    TOML Configuration Example:
        [workflow]
        mode = "single"           # Execution mode: "single", "autonomous", or "batch"
        auto_validate = true      # Automatically run validation after task completion
        journal_enabled = true    # Enable journaling of task completions
        batch_size = 5            # Number of tasks to execute in batch mode
        context_threshold = 85    # Context usage threshold (%) to trigger pause

    Environment Variables:
        - FOUNDRY_MCP_WORKFLOW_MODE: Workflow execution mode
        - FOUNDRY_MCP_WORKFLOW_AUTO_VALIDATE: Enable auto-validation (true/false)
        - FOUNDRY_MCP_WORKFLOW_JOURNAL_ENABLED: Enable journaling (true/false)
        - FOUNDRY_MCP_WORKFLOW_BATCH_SIZE: Batch size for batch mode
        - FOUNDRY_MCP_WORKFLOW_CONTEXT_THRESHOLD: Context threshold percentage

    Attributes:
        mode: Workflow execution mode
        auto_validate: Whether to run validation after task completion
        journal_enabled: Whether to journal task completions
        batch_size: Number of tasks to execute in batch mode
        context_threshold: Context usage threshold to trigger pause (percentage)
    """

    mode: WorkflowMode = WorkflowMode.SINGLE
    auto_validate: bool = True
    journal_enabled: bool = True
    batch_size: int = 5
    context_threshold: int = 85

    def validate(self) -> None:
        """Validate the workflow configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

        if not 50 <= self.context_threshold <= 100:
            raise ValueError(
                f"context_threshold must be between 50 and 100, got {self.context_threshold}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        """Create WorkflowConfig from a dictionary (typically the [workflow] section).

        Args:
            data: Dictionary with workflow configuration values

        Returns:
            WorkflowConfig instance

        Raises:
            ValueError: If mode is invalid
        """
        config = cls()

        # Parse mode
        if "mode" in data:
            mode_str = data["mode"].lower()
            try:
                config.mode = WorkflowMode(mode_str)
            except ValueError:
                valid = [m.value for m in WorkflowMode]
                raise ValueError(
                    f"Invalid workflow mode '{mode_str}'. Must be one of: {valid}"
                )

        # Parse boolean fields
        if "auto_validate" in data:
            config.auto_validate = bool(data["auto_validate"])

        if "journal_enabled" in data:
            config.journal_enabled = bool(data["journal_enabled"])

        # Parse integer fields
        if "batch_size" in data:
            config.batch_size = int(data["batch_size"])

        if "context_threshold" in data:
            config.context_threshold = int(data["context_threshold"])

        return config

    @classmethod
    def from_toml(cls, path: Path) -> "WorkflowConfig":
        """Load workflow configuration from a TOML file.

        Args:
            path: Path to the TOML configuration file

        Returns:
            WorkflowConfig instance with parsed settings

        Raises:
            FileNotFoundError: If the config file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data.get("workflow", {}))

    @classmethod
    def from_env(cls) -> "WorkflowConfig":
        """Create WorkflowConfig from environment variables only.

        Environment variables:
            - FOUNDRY_MCP_WORKFLOW_MODE: Workflow execution mode
            - FOUNDRY_MCP_WORKFLOW_AUTO_VALIDATE: Enable auto-validation
            - FOUNDRY_MCP_WORKFLOW_JOURNAL_ENABLED: Enable journaling
            - FOUNDRY_MCP_WORKFLOW_BATCH_SIZE: Batch size
            - FOUNDRY_MCP_WORKFLOW_CONTEXT_THRESHOLD: Context threshold

        Returns:
            WorkflowConfig instance with environment-based settings
        """
        config = cls()

        # Mode
        if mode := os.environ.get("FOUNDRY_MCP_WORKFLOW_MODE"):
            try:
                config.mode = WorkflowMode(mode.lower())
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_WORKFLOW_MODE: {mode}, using default")

        # Auto-validate
        if auto_validate := os.environ.get("FOUNDRY_MCP_WORKFLOW_AUTO_VALIDATE"):
            config.auto_validate = auto_validate.lower() in ("true", "1", "yes")

        # Journal enabled
        if journal := os.environ.get("FOUNDRY_MCP_WORKFLOW_JOURNAL_ENABLED"):
            config.journal_enabled = journal.lower() in ("true", "1", "yes")

        # Batch size
        if batch_size := os.environ.get("FOUNDRY_MCP_WORKFLOW_BATCH_SIZE"):
            try:
                config.batch_size = int(batch_size)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_WORKFLOW_BATCH_SIZE: {batch_size}, using default")

        # Context threshold
        if threshold := os.environ.get("FOUNDRY_MCP_WORKFLOW_CONTEXT_THRESHOLD"):
            try:
                config.context_threshold = int(threshold)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_WORKFLOW_CONTEXT_THRESHOLD: {threshold}, using default")

        return config


def load_workflow_config(
    config_file: Optional[Path] = None,
    use_env_fallback: bool = True,
) -> WorkflowConfig:
    """Load workflow configuration from TOML file with environment fallback.

    Priority (highest to lowest):
    1. TOML config file (if provided or found at default locations)
    2. Environment variables
    3. Default values

    Args:
        config_file: Optional path to TOML config file
        use_env_fallback: Whether to use environment variables as fallback

    Returns:
        WorkflowConfig instance with merged settings
    """
    config = WorkflowConfig()

    # Try to load from TOML
    toml_loaded = False
    if config_file and config_file.exists():
        try:
            config = WorkflowConfig.from_toml(config_file)
            toml_loaded = True
            logger.debug(f"Loaded workflow config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load workflow config from {config_file}: {e}")
    else:
        # Try default locations
        default_paths = [
            Path("foundry-mcp.toml"),
            Path(".foundry-mcp.toml"),
            Path.home() / ".config" / "foundry-mcp" / "config.toml",
        ]
        for path in default_paths:
            if path.exists():
                try:
                    config = WorkflowConfig.from_toml(path)
                    toml_loaded = True
                    logger.debug(f"Loaded workflow config from {path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load from {path}: {e}")

    # Apply environment variable overrides
    if use_env_fallback:
        env_config = WorkflowConfig.from_env()

        # Mode override
        if not toml_loaded or config.mode == WorkflowMode.SINGLE:
            if os.environ.get("FOUNDRY_MCP_WORKFLOW_MODE"):
                config.mode = env_config.mode

        # Boolean overrides (env can override TOML)
        if os.environ.get("FOUNDRY_MCP_WORKFLOW_AUTO_VALIDATE"):
            config.auto_validate = env_config.auto_validate

        if os.environ.get("FOUNDRY_MCP_WORKFLOW_JOURNAL_ENABLED"):
            config.journal_enabled = env_config.journal_enabled

        # Integer overrides
        if config.batch_size == 5 and env_config.batch_size != 5:
            config.batch_size = env_config.batch_size

        if config.context_threshold == 85 and env_config.context_threshold != 85:
            config.context_threshold = env_config.context_threshold

    return config


# Global workflow configuration instance
_workflow_config: Optional[WorkflowConfig] = None


def get_workflow_config() -> WorkflowConfig:
    """Get the global workflow configuration instance.

    Returns:
        WorkflowConfig instance (loaded from file/env on first call)
    """
    global _workflow_config
    if _workflow_config is None:
        _workflow_config = load_workflow_config()
    return _workflow_config


def set_workflow_config(config: WorkflowConfig) -> None:
    """Set the global workflow configuration instance.

    Args:
        config: WorkflowConfig instance to use globally
    """
    global _workflow_config
    _workflow_config = config


def reset_workflow_config() -> None:
    """Reset the global workflow configuration to None.

    Useful for testing or reloading configuration.
    """
    global _workflow_config
    _workflow_config = None
