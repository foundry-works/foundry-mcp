"""Consultation configuration: ConsultationConfig and load/get/set/reset functions."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

from .provider_spec import ProviderSpec
from ._paths import _default_config_search_paths

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConsultationConfig:
    """Per-workflow consultation configuration overrides.

    Allows individual workflows to specify minimum model requirements,
    timeout overrides, and default review types for AI consultations.

    TOML Configuration Example:
        [consultation.workflows.fidelity_review]
        min_models = 2
        timeout_override = 600.0
        default_review_type = "full"

        [consultation.workflows.plan_review]
        min_models = 3
        default_review_type = "full"

    Attributes:
        min_models: Minimum number of models required for consensus (default: 1).
                    When set > 1, the consultation orchestrator will gather
                    responses from multiple providers before synthesizing.
        timeout_override: Optional timeout override in seconds. When set,
                          overrides the default_timeout from ConsultationConfig
                          for this specific workflow.
        default_review_type: Default review type for this workflow (default: "full").
                             Valid values: "quick", "full", "security", "feasibility".
                             Used when no explicit review_type is provided in requests.
    """

    min_models: int = 1
    timeout_override: Optional[float] = None
    default_review_type: str = "full"

    # Valid review types
    VALID_REVIEW_TYPES = {"quick", "full", "security", "feasibility"}

    def validate(self) -> None:
        """Validate the workflow consultation configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.min_models < 1:
            raise ValueError(f"min_models must be at least 1, got {self.min_models}")

        if self.timeout_override is not None and self.timeout_override <= 0:
            raise ValueError(
                f"timeout_override must be positive if set, got {self.timeout_override}"
            )

        if self.default_review_type not in self.VALID_REVIEW_TYPES:
            raise ValueError(
                f"default_review_type must be one of {sorted(self.VALID_REVIEW_TYPES)}, "
                f"got '{self.default_review_type}'"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConsultationConfig":
        """Create WorkflowConsultationConfig from a dictionary.

        Args:
            data: Dictionary with workflow consultation configuration values

        Returns:
            WorkflowConsultationConfig instance
        """
        config = cls()

        if "min_models" in data:
            config.min_models = int(data["min_models"])

        if "timeout_override" in data:
            value = data["timeout_override"]
            if value is not None:
                config.timeout_override = float(value)

        if "default_review_type" in data:
            config.default_review_type = str(data["default_review_type"]).lower()

        return config


@dataclass
class ConsultationConfig:
    """AI consultation configuration parsed from foundry-mcp.toml [consultation] section.

    TOML Configuration Example:
        [consultation]
        # Provider priority list - first available wins
        # Format: "[api]provider/model" or "[cli]transport[:backend/model|:model]"
        priority = [
            "[cli]gemini:pro",
            "[cli]claude:opus",
            "[cli]opencode:openai/gpt-5.2",
            "[api]openai/gpt-4.1",
        ]

        # Per-provider overrides (optional)
        [consultation.overrides]
        "[cli]opencode:openai/gpt-5.2" = { timeout = 600 }
        "[api]openai/gpt-4.1" = { temperature = 0.3 }

        # Operational settings
        default_timeout = 300       # Default timeout in seconds (default: 300)
        max_retries = 2             # Max retry attempts on failure (default: 2)
        retry_delay = 5.0           # Delay between retries in seconds (default: 5.0)
        fallback_enabled = true     # Enable fallback to next provider (default: true)
        cache_ttl = 3600            # Cache TTL in seconds (default: 3600)

        # Per-workflow configuration (optional)
        [consultation.workflows.fidelity_review]
        min_models = 2              # Require 2 models for consensus
        timeout_override = 600.0    # Override default timeout

        [consultation.workflows.plan_review]
        min_models = 3              # Require 3 models for plan reviews

    Environment Variables:
        - FOUNDRY_MCP_CONSULTATION_TIMEOUT: Default timeout
        - FOUNDRY_MCP_CONSULTATION_MAX_RETRIES: Max retry attempts
        - FOUNDRY_MCP_CONSULTATION_RETRY_DELAY: Delay between retries
        - FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED: Enable provider fallback
        - FOUNDRY_MCP_CONSULTATION_CACHE_TTL: Cache TTL
        - FOUNDRY_MCP_CONSULTATION_PRIORITY: Comma-separated priority list

    Attributes:
        priority: List of provider specs in priority order (first available wins)
        overrides: Per-provider setting overrides (keyed by spec string)
        default_timeout: Default timeout for AI consultations in seconds
        max_retries: Maximum retry attempts on transient failures
        retry_delay: Delay between retry attempts in seconds
        fallback_enabled: Whether to try next provider on failure
        cache_ttl: Time-to-live for cached consultation results in seconds
        workflows: Per-workflow configuration overrides (keyed by workflow name)
    """

    priority: List[str] = field(default_factory=list)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_timeout: float = 300.0
    max_retries: int = 2
    retry_delay: float = 5.0
    fallback_enabled: bool = True
    cache_ttl: int = 3600
    workflows: Dict[str, WorkflowConsultationConfig] = field(default_factory=dict)

    def get_provider_specs(self) -> List[ProviderSpec]:
        """Parse priority list into ProviderSpec objects.

        Returns:
            List of parsed ProviderSpec instances

        Raises:
            ValueError: If any spec in priority list is invalid
        """
        return [ProviderSpec.parse(spec) for spec in self.priority]

    def get_override(self, spec: str) -> Dict[str, Any]:
        """Get override settings for a specific provider spec.

        Args:
            spec: Provider spec string (e.g., "[api]openai/gpt-4.1")

        Returns:
            Override dictionary (empty if no overrides configured)
        """
        return self.overrides.get(spec, {})

    def get_workflow_config(self, workflow_name: str) -> WorkflowConsultationConfig:
        """Get configuration for a specific workflow.

        Args:
            workflow_name: Name of the workflow (e.g., "fidelity_review", "plan_review")

        Returns:
            WorkflowConsultationConfig for the workflow. Returns a default instance
            with min_models=1 if no workflow-specific config exists.

        Examples:
            >>> config = ConsultationConfig()
            >>> config.workflows["fidelity_review"] = WorkflowConsultationConfig(min_models=2)
            >>> fidelity = config.get_workflow_config("fidelity_review")
            >>> fidelity.min_models
            2
            >>> unknown = config.get_workflow_config("unknown_workflow")
            >>> unknown.min_models
            1
        """
        return self.workflows.get(workflow_name, WorkflowConsultationConfig())

    def validate(self) -> None:
        """Validate the consultation configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.default_timeout <= 0:
            raise ValueError(f"default_timeout must be positive, got {self.default_timeout}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be non-negative, got {self.retry_delay}")

        if self.cache_ttl <= 0:
            raise ValueError(f"cache_ttl must be positive, got {self.cache_ttl}")

        # Validate priority list
        all_errors = []
        for spec_str in self.priority:
            try:
                spec = ProviderSpec.parse(spec_str)
                errors = spec.validate()
                if errors:
                    all_errors.extend([f"{spec_str}: {e}" for e in errors])
            except ValueError as e:
                all_errors.append(f"{spec_str}: {e}")

        if all_errors:
            raise ValueError("Invalid provider specs in priority list:\n" + "\n".join(all_errors))

        # Validate workflow configurations
        workflow_errors = []
        for workflow_name, workflow_config in self.workflows.items():
            try:
                workflow_config.validate()
            except ValueError as e:
                workflow_errors.append(f"workflows.{workflow_name}: {e}")

        if workflow_errors:
            raise ValueError("Invalid workflow configurations:\n" + "\n".join(workflow_errors))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsultationConfig":
        """Create ConsultationConfig from a dictionary (typically the [consultation] section).

        Args:
            data: Dictionary with consultation configuration values

        Returns:
            ConsultationConfig instance
        """
        config = cls()

        # Parse priority list
        if "priority" in data:
            priority = data["priority"]
            if isinstance(priority, list):
                config.priority = [str(p) for p in priority]
            else:
                logger.warning(f"Invalid priority format (expected list): {type(priority)}")

        # Parse overrides
        if "overrides" in data:
            overrides = data["overrides"]
            if isinstance(overrides, dict):
                config.overrides = {str(k): dict(v) for k, v in overrides.items()}
            else:
                logger.warning(f"Invalid overrides format (expected dict): {type(overrides)}")

        if "default_timeout" in data:
            config.default_timeout = float(data["default_timeout"])

        if "max_retries" in data:
            config.max_retries = int(data["max_retries"])

        if "retry_delay" in data:
            config.retry_delay = float(data["retry_delay"])

        if "fallback_enabled" in data:
            config.fallback_enabled = bool(data["fallback_enabled"])

        if "cache_ttl" in data:
            config.cache_ttl = int(data["cache_ttl"])

        # Parse workflow configurations
        if "workflows" in data:
            workflows = data["workflows"]
            if isinstance(workflows, dict):
                for workflow_name, workflow_data in workflows.items():
                    if isinstance(workflow_data, dict):
                        config.workflows[str(workflow_name)] = (
                            WorkflowConsultationConfig.from_dict(workflow_data)
                        )
                    else:
                        logger.warning(
                            f"Invalid workflow config format for '{workflow_name}' "
                            f"(expected dict): {type(workflow_data)}"
                        )
            else:
                logger.warning(f"Invalid workflows format (expected dict): {type(workflows)}")

        return config

    @classmethod
    def from_toml(cls, path: Path) -> "ConsultationConfig":
        """Load consultation configuration from a TOML file.

        Args:
            path: Path to the TOML configuration file

        Returns:
            ConsultationConfig instance with parsed settings

        Raises:
            FileNotFoundError: If the config file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data.get("consultation", {}))

    @classmethod
    def from_env(cls) -> "ConsultationConfig":
        """Create ConsultationConfig from environment variables only.

        Environment variables:
            - FOUNDRY_MCP_CONSULTATION_PRIORITY: Comma-separated priority list
            - FOUNDRY_MCP_CONSULTATION_TIMEOUT: Default timeout in seconds
            - FOUNDRY_MCP_CONSULTATION_MAX_RETRIES: Max retry attempts
            - FOUNDRY_MCP_CONSULTATION_RETRY_DELAY: Delay between retries
            - FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED: Enable fallback (true/false)
            - FOUNDRY_MCP_CONSULTATION_CACHE_TTL: Cache TTL in seconds

        Returns:
            ConsultationConfig instance with environment-based settings
        """
        config = cls()

        # Priority list (comma-separated)
        if priority := os.environ.get("FOUNDRY_MCP_CONSULTATION_PRIORITY"):
            config.priority = [p.strip() for p in priority.split(",") if p.strip()]

        # Timeout
        if timeout := os.environ.get("FOUNDRY_MCP_CONSULTATION_TIMEOUT"):
            try:
                config.default_timeout = float(timeout)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_CONSULTATION_TIMEOUT: {timeout}, using default")

        # Max retries
        if max_retries := os.environ.get("FOUNDRY_MCP_CONSULTATION_MAX_RETRIES"):
            try:
                config.max_retries = int(max_retries)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_CONSULTATION_MAX_RETRIES: {max_retries}, using default")

        # Retry delay
        if retry_delay := os.environ.get("FOUNDRY_MCP_CONSULTATION_RETRY_DELAY"):
            try:
                config.retry_delay = float(retry_delay)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_CONSULTATION_RETRY_DELAY: {retry_delay}, using default")

        # Fallback enabled
        if fallback := os.environ.get("FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED"):
            config.fallback_enabled = fallback.lower() in ("true", "1", "yes")

        # Cache TTL
        if cache_ttl := os.environ.get("FOUNDRY_MCP_CONSULTATION_CACHE_TTL"):
            try:
                config.cache_ttl = int(cache_ttl)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_CONSULTATION_CACHE_TTL: {cache_ttl}, using default")

        return config


def load_consultation_config(
    config_file: Optional[Path] = None,
    use_env_fallback: bool = True,
) -> ConsultationConfig:
    """Load consultation configuration from TOML file with environment fallback.

    Priority (highest to lowest):
    1. TOML config file (if provided or found at default locations)
    2. Environment variables
    3. Default values

    Args:
        config_file: Optional path to TOML config file
        use_env_fallback: Whether to use environment variables as fallback

    Returns:
        ConsultationConfig instance with merged settings
    """
    config = ConsultationConfig()

    # Try to load from TOML
    if config_file and config_file.exists():
        try:
            config = ConsultationConfig.from_toml(config_file)
            logger.debug(f"Loaded consultation config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load consultation config from {config_file}: {e}")
    else:
        # Try default locations (lowest to highest priority — last match wins)
        for path in _default_config_search_paths():
            if path.exists():
                try:
                    config = ConsultationConfig.from_toml(path)
                    logger.debug(f"Loaded consultation config from {path}")
                except Exception as e:
                    logger.debug(f"Failed to load from {path}: {e}")

    # Apply environment variable overrides
    if use_env_fallback:
        env_config = ConsultationConfig.from_env()

        # Priority override (env can override TOML if set)
        if not config.priority and env_config.priority:
            config.priority = env_config.priority
        elif os.environ.get("FOUNDRY_MCP_CONSULTATION_PRIORITY"):
            # Explicit env var overrides TOML
            config.priority = env_config.priority

        # Timeout override
        if config.default_timeout == 300.0 and env_config.default_timeout != 300.0:
            config.default_timeout = env_config.default_timeout

        # Max retries override
        if config.max_retries == 2 and env_config.max_retries != 2:
            config.max_retries = env_config.max_retries

        # Retry delay override
        if config.retry_delay == 5.0 and env_config.retry_delay != 5.0:
            config.retry_delay = env_config.retry_delay

        # Fallback enabled (env can override TOML)
        if os.environ.get("FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED"):
            config.fallback_enabled = env_config.fallback_enabled

        # Cache TTL override
        if config.cache_ttl == 3600 and env_config.cache_ttl != 3600:
            config.cache_ttl = env_config.cache_ttl

    return config


# Global consultation configuration instance
_consultation_config: Optional[ConsultationConfig] = None


def get_consultation_config() -> ConsultationConfig:
    """Get the global consultation configuration instance.

    Returns:
        ConsultationConfig instance (loaded from file/env on first call)
    """
    global _consultation_config
    if _consultation_config is None:
        _consultation_config = load_consultation_config()
    return _consultation_config


def set_consultation_config(config: ConsultationConfig) -> None:
    """Set the global consultation configuration instance.

    Args:
        config: ConsultationConfig instance to use globally
    """
    global _consultation_config
    _consultation_config = config


def reset_consultation_config() -> None:
    """Reset the global consultation configuration to None.

    Useful for testing or reloading configuration.
    """
    global _consultation_config
    _consultation_config = None
