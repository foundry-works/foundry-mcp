"""LLM configuration: LLMConfig class and load/get/set/reset functions."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

from .provider_spec import LLMProviderType, DEFAULT_MODELS, API_KEY_ENV_VARS
from ._paths import _default_config_search_paths

logger = logging.getLogger(__name__)


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
        # Try default locations (lowest to highest priority — last match wins)
        for path in _default_config_search_paths():
            if path.exists():
                try:
                    loaded = LLMConfig.from_toml(path)
                    # Only use if the file actually has an [llm] section
                    # (check if it differs from bare defaults)
                    config = loaded
                    toml_loaded = True
                    logger.debug(f"Loaded LLM config from {path}")
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
