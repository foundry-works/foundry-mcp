"""
LLM configuration parsing for foundry-mcp.

Parses the [llm] section from foundry-mcp.toml to configure LLM provider settings.

TOML Configuration Example:
    [llm]
    provider = "openai"           # Required: "openai", "anthropic", or "local"
    api_key = "sk-..."            # Optional: defaults to env var based on provider
    model = "gpt-4.1"             # Optional: provider-specific default
    timeout = 30                  # Optional: request timeout in seconds (default: 30)

Environment Variables (fallback if not in TOML):
    - FOUNDRY_MCP_LLM_PROVIDER: LLM provider type ("openai", "anthropic", "local")
    - FOUNDRY_MCP_LLM_API_KEY: API key (takes precedence over provider-specific keys)
    - FOUNDRY_MCP_LLM_MODEL: Model identifier
    - FOUNDRY_MCP_LLM_TIMEOUT: Request timeout in seconds
    - FOUNDRY_MCP_LLM_BASE_URL: Custom API base URL
    - FOUNDRY_MCP_LLM_MAX_TOKENS: Default max tokens
    - FOUNDRY_MCP_LLM_TEMPERATURE: Default temperature
    - FOUNDRY_MCP_LLM_ORGANIZATION: Organization ID (OpenAI)

Provider-specific API key fallbacks:
    - OPENAI_API_KEY: OpenAI API key (if FOUNDRY_MCP_LLM_API_KEY not set)
    - ANTHROPIC_API_KEY: Anthropic API key (if FOUNDRY_MCP_LLM_API_KEY not set)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback


logger = logging.getLogger(__name__)


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


# Default models per provider
DEFAULT_MODELS: Dict[LLMProviderType, str] = {
    LLMProviderType.OPENAI: "gpt-4.1",
    LLMProviderType.ANTHROPIC: "claude-sonnet-4-5",
    LLMProviderType.LOCAL: "llama4",
}

# Environment variable names for API keys
API_KEY_ENV_VARS: Dict[LLMProviderType, str] = {
    LLMProviderType.OPENAI: "OPENAI_API_KEY",
    LLMProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProviderType.LOCAL: "",  # Local providers typically don't need keys
}


@dataclass
class LLMConfig:
    """LLM configuration parsed from foundry-mcp.toml.

    Attributes:
        provider: The LLM provider type ("openai", "anthropic", "local")
        api_key: API key for the provider (optional, falls back to env var)
        model: Model identifier (optional, uses provider default)
        timeout: Request timeout in seconds (default: 30)
        base_url: Custom API base URL (optional, for proxies or local servers)
        organization: Organization ID (OpenAI only)
        max_tokens: Default max tokens for responses
        temperature: Default temperature for generation
    """

    provider: LLMProviderType = LLMProviderType.OPENAI
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 30
    base_url: Optional[str] = None
    organization: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7

    def get_api_key(self) -> Optional[str]:
        """Get API key, falling back to environment variables if not set.

        Priority:
        1. Explicit api_key set in config
        2. FOUNDRY_MCP_LLM_API_KEY environment variable
        3. Provider-specific env var (OPENAI_API_KEY, ANTHROPIC_API_KEY)

        Returns:
            API key string or None if not available
        """
        if self.api_key:
            return self.api_key

        # Check unified env var first
        if unified_key := os.environ.get("FOUNDRY_MCP_LLM_API_KEY"):
            return unified_key

        # Fall back to provider-specific env var
        env_var = API_KEY_ENV_VARS.get(self.provider, "")
        if env_var:
            return os.environ.get(env_var)

        return None

    def get_model(self) -> str:
        """Get model, falling back to provider default if not set.

        Returns:
            Model identifier string
        """
        if self.model:
            return self.model

        return DEFAULT_MODELS.get(self.provider, "gpt-4.1")

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

        # Validate max_tokens
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        # Validate temperature
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")

        # Check API key for non-local providers
        if self.provider != LLMProviderType.LOCAL:
            if not self.get_api_key():
                env_var = API_KEY_ENV_VARS.get(self.provider, "")
                raise ValueError(
                    f"API key required for {self.provider.value} provider. "
                    f"Set 'api_key' in config or {env_var} environment variable."
                )

    @classmethod
    def from_toml(cls, path: Path) -> "LLMConfig":
        """Load LLM configuration from a TOML file.

        Args:
            path: Path to the TOML configuration file

        Returns:
            LLMConfig instance with parsed settings

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the configuration is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data.get("llm", {}))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from a dictionary (typically the [llm] section).

        Args:
            data: Dictionary with LLM configuration values

        Returns:
            LLMConfig instance

        Raises:
            ValueError: If provider type is invalid
        """
        config = cls()

        # Parse provider
        if "provider" in data:
            provider_str = data["provider"].lower()
            try:
                config.provider = LLMProviderType(provider_str)
            except ValueError:
                valid = [p.value for p in LLMProviderType]
                raise ValueError(
                    f"Invalid provider '{provider_str}'. Must be one of: {valid}"
                )

        # Parse other fields
        if "api_key" in data:
            config.api_key = data["api_key"]

        if "model" in data:
            config.model = data["model"]

        if "timeout" in data:
            config.timeout = int(data["timeout"])

        if "base_url" in data:
            config.base_url = data["base_url"]

        if "organization" in data:
            config.organization = data["organization"]

        if "max_tokens" in data:
            config.max_tokens = int(data["max_tokens"])

        if "temperature" in data:
            config.temperature = float(data["temperature"])

        return config

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables only.

        Environment variables:
            - FOUNDRY_MCP_LLM_PROVIDER: Provider type ("openai", "anthropic", "local")
            - FOUNDRY_MCP_LLM_API_KEY: API key (unified, takes precedence)
            - FOUNDRY_MCP_LLM_MODEL: Model identifier
            - FOUNDRY_MCP_LLM_TIMEOUT: Request timeout in seconds
            - FOUNDRY_MCP_LLM_BASE_URL: Custom API base URL
            - FOUNDRY_MCP_LLM_MAX_TOKENS: Default max tokens
            - FOUNDRY_MCP_LLM_TEMPERATURE: Default temperature
            - FOUNDRY_MCP_LLM_ORGANIZATION: Organization ID (OpenAI only)

        Returns:
            LLMConfig instance with environment-based settings
        """
        config = cls()

        # Provider
        if provider := os.environ.get("FOUNDRY_MCP_LLM_PROVIDER"):
            try:
                config.provider = LLMProviderType(provider.lower())
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_LLM_PROVIDER: {provider}, using default")

        # API Key (explicit env var, not the provider-specific fallback)
        if api_key := os.environ.get("FOUNDRY_MCP_LLM_API_KEY"):
            config.api_key = api_key

        # Model
        if model := os.environ.get("FOUNDRY_MCP_LLM_MODEL"):
            config.model = model

        # Timeout
        if timeout := os.environ.get("FOUNDRY_MCP_LLM_TIMEOUT"):
            try:
                config.timeout = int(timeout)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_LLM_TIMEOUT: {timeout}, using default")

        # Base URL
        if base_url := os.environ.get("FOUNDRY_MCP_LLM_BASE_URL"):
            config.base_url = base_url

        # Max tokens
        if max_tokens := os.environ.get("FOUNDRY_MCP_LLM_MAX_TOKENS"):
            try:
                config.max_tokens = int(max_tokens)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_LLM_MAX_TOKENS: {max_tokens}, using default")

        # Temperature
        if temperature := os.environ.get("FOUNDRY_MCP_LLM_TEMPERATURE"):
            try:
                config.temperature = float(temperature)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_MCP_LLM_TEMPERATURE: {temperature}, using default")

        # Organization
        if organization := os.environ.get("FOUNDRY_MCP_LLM_ORGANIZATION"):
            config.organization = organization

        return config


def load_llm_config(
    config_file: Optional[Path] = None,
    use_env_fallback: bool = True,
) -> LLMConfig:
    """Load LLM configuration from TOML file with environment fallback.

    Priority (highest to lowest):
    1. TOML config file (if provided or found at default locations)
    2. Environment variables
    3. Default values

    Args:
        config_file: Optional path to TOML config file
        use_env_fallback: Whether to use environment variables as fallback

    Returns:
        LLMConfig instance with merged settings
    """
    config = LLMConfig()

    # Try to load from TOML
    toml_loaded = False
    if config_file and config_file.exists():
        try:
            config = LLMConfig.from_toml(config_file)
            toml_loaded = True
            logger.debug(f"Loaded LLM config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load LLM config from {config_file}: {e}")
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
                    config = LLMConfig.from_toml(path)
                    toml_loaded = True
                    logger.debug(f"Loaded LLM config from {path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load from {path}: {e}")

    # Apply environment variable overrides
    if use_env_fallback:
        env_config = LLMConfig.from_env()

        # Only override if TOML didn't set the value
        if not toml_loaded or config.provider == LLMProviderType.OPENAI:
            if os.environ.get("FOUNDRY_MCP_LLM_PROVIDER"):
                config.provider = env_config.provider

        # API key: env overrides if set (explicit FOUNDRY_MCP_LLM_API_KEY)
        if not config.api_key and env_config.api_key:
            config.api_key = env_config.api_key

        if not config.model and env_config.model:
            config.model = env_config.model

        if config.timeout == 30 and env_config.timeout != 30:
            config.timeout = env_config.timeout

        if not config.base_url and env_config.base_url:
            config.base_url = env_config.base_url

        if config.max_tokens == 1024 and env_config.max_tokens != 1024:
            config.max_tokens = env_config.max_tokens

        if config.temperature == 0.7 and env_config.temperature != 0.7:
            config.temperature = env_config.temperature

        if not config.organization and env_config.organization:
            config.organization = env_config.organization

    return config


# Global configuration instance
_llm_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration instance.

    Returns:
        LLMConfig instance (loaded from file/env on first call)
    """
    global _llm_config
    if _llm_config is None:
        _llm_config = load_llm_config()
    return _llm_config


def set_llm_config(config: LLMConfig) -> None:
    """Set the global LLM configuration instance.

    Args:
        config: LLMConfig instance to use globally
    """
    global _llm_config
    _llm_config = config


def reset_llm_config() -> None:
    """Reset the global LLM configuration to None.

    Useful for testing or reloading configuration.
    """
    global _llm_config
    _llm_config = None


# =============================================================================
# Workflow Configuration
# =============================================================================


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


__all__ = [
    # LLM Config
    "LLMProviderType",
    "LLMConfig",
    "load_llm_config",
    "get_llm_config",
    "set_llm_config",
    "reset_llm_config",
    "DEFAULT_MODELS",
    "API_KEY_ENV_VARS",
    # Workflow Config
    "WorkflowMode",
    "WorkflowConfig",
    "load_workflow_config",
    "get_workflow_config",
    "set_workflow_config",
    "reset_workflow_config",
]
