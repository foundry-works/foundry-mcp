"""
LLM configuration parsing for foundry-mcp.

Parses the [llm] section from foundry-mcp.toml to configure LLM provider settings.

TOML Configuration Example:
    [llm]
    provider = "openai"           # Required: "openai", "anthropic", or "local"
    api_key = "sk-..."            # Optional: defaults to env var based on provider
    model = "gpt-4"               # Optional: provider-specific default
    timeout = 30                  # Optional: request timeout in seconds (default: 30)

Environment Variables (fallback if not in TOML):
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - FOUNDRY_LLM_PROVIDER: Default provider
    - FOUNDRY_LLM_MODEL: Default model
    - FOUNDRY_LLM_TIMEOUT: Request timeout
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
    LLMProviderType.OPENAI: "gpt-4",
    LLMProviderType.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProviderType.LOCAL: "llama3.2",
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
        """Get API key, falling back to environment variable if not set.

        Returns:
            API key string or None if not available
        """
        if self.api_key:
            return self.api_key

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

        return DEFAULT_MODELS.get(self.provider, "gpt-4")

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
            - FOUNDRY_LLM_PROVIDER: Provider type
            - FOUNDRY_LLM_MODEL: Model identifier
            - FOUNDRY_LLM_TIMEOUT: Request timeout
            - FOUNDRY_LLM_BASE_URL: Custom API base URL
            - FOUNDRY_LLM_MAX_TOKENS: Default max tokens
            - FOUNDRY_LLM_TEMPERATURE: Default temperature

        Returns:
            LLMConfig instance with environment-based settings
        """
        config = cls()

        # Provider
        if provider := os.environ.get("FOUNDRY_LLM_PROVIDER"):
            try:
                config.provider = LLMProviderType(provider.lower())
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_LLM_PROVIDER: {provider}, using default")

        # Model
        if model := os.environ.get("FOUNDRY_LLM_MODEL"):
            config.model = model

        # Timeout
        if timeout := os.environ.get("FOUNDRY_LLM_TIMEOUT"):
            try:
                config.timeout = int(timeout)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_LLM_TIMEOUT: {timeout}, using default")

        # Base URL
        if base_url := os.environ.get("FOUNDRY_LLM_BASE_URL"):
            config.base_url = base_url

        # Max tokens
        if max_tokens := os.environ.get("FOUNDRY_LLM_MAX_TOKENS"):
            try:
                config.max_tokens = int(max_tokens)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_LLM_MAX_TOKENS: {max_tokens}, using default")

        # Temperature
        if temperature := os.environ.get("FOUNDRY_LLM_TEMPERATURE"):
            try:
                config.temperature = float(temperature)
            except ValueError:
                logger.warning(f"Invalid FOUNDRY_LLM_TEMPERATURE: {temperature}, using default")

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
            if os.environ.get("FOUNDRY_LLM_PROVIDER"):
                config.provider = env_config.provider

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


__all__ = [
    "LLMProviderType",
    "LLMConfig",
    "load_llm_config",
    "get_llm_config",
    "set_llm_config",
    "reset_llm_config",
    "DEFAULT_MODELS",
    "API_KEY_ENV_VARS",
]
