"""Provider specification (unified priority notation) and LLM provider types."""

import re
from enum import Enum
from typing import Optional, Dict, List, Literal
from dataclasses import dataclass


@dataclass
class ProviderSpec:
    """Parsed provider specification from hybrid notation.

    Supports bracket-prefix notation for unified API/CLI provider configuration:
        - [api]openai/gpt-4.1           -> API provider with model
        - [api]anthropic/claude-sonnet-4 -> API provider with model
        - [cli]gemini:pro               -> CLI provider with model
        - [cli]claude:opus              -> CLI provider with model
        - [cli]opencode:openai/gpt-5.2  -> CLI provider routing to backend
        - [cli]codex                    -> CLI provider with default model

    Grammar:
        spec       := "[api]" api_spec | "[cli]" cli_spec
        api_spec   := provider "/" model
        cli_spec   := transport (":" backend "/" model | ":" model | "")

    Attributes:
        type: Provider type - "api" for direct API calls, "cli" for CLI tools
        provider: Provider/transport identifier (openai, gemini, opencode, etc.)
        backend: Optional backend for CLI routing (openai, anthropic, gemini)
        model: Optional model identifier (gpt-4.1, pro, opus, etc.)
        raw: Original specification string for error messages
    """

    type: Literal["api", "cli"]
    provider: str
    backend: Optional[str] = None
    model: Optional[str] = None
    raw: str = ""

    # Known providers for validation
    KNOWN_API_PROVIDERS = {"openai", "anthropic", "local"}
    KNOWN_CLI_PROVIDERS = {"gemini", "codex", "cursor-agent", "opencode", "claude"}
    KNOWN_BACKENDS = {"openai", "anthropic", "gemini", "local"}

    # Regex patterns for parsing
    _API_PATTERN = re.compile(r"^\[api\]([^/]+)/(.+)$")
    _CLI_FULL_PATTERN = re.compile(r"^\[cli\]([^:]+):([^/]+)/(.+)$")  # transport:backend/model
    _CLI_MODEL_PATTERN = re.compile(r"^\[cli\]([^:]+):([^/]+)$")  # transport:model
    _CLI_SIMPLE_PATTERN = re.compile(r"^\[cli\]([^:]+)$")  # transport only

    @classmethod
    def parse(cls, spec: str) -> "ProviderSpec":
        """Parse a provider specification string.

        Args:
            spec: Provider spec in bracket notation (e.g., "[api]openai/gpt-4.1")

        Returns:
            ProviderSpec instance with parsed components

        Raises:
            ValueError: If the spec format is invalid

        Examples:
            >>> ProviderSpec.parse("[api]openai/gpt-4.1")
            ProviderSpec(type='api', provider='openai', model='gpt-4.1')

            >>> ProviderSpec.parse("[cli]gemini:pro")
            ProviderSpec(type='cli', provider='gemini', model='pro')

            >>> ProviderSpec.parse("[cli]opencode:openai/gpt-5.2")
            ProviderSpec(type='cli', provider='opencode', backend='openai', model='gpt-5.2')
        """
        spec = spec.strip()

        if not spec:
            raise ValueError("Provider spec cannot be empty")

        # Try API pattern: [api]provider/model
        if match := cls._API_PATTERN.match(spec):
            provider, model = match.groups()
            return cls(
                type="api",
                provider=provider.lower(),
                model=model,
                raw=spec,
            )

        # Try CLI full pattern: [cli]transport:backend/model
        if match := cls._CLI_FULL_PATTERN.match(spec):
            transport, backend, model = match.groups()
            return cls(
                type="cli",
                provider=transport.lower(),
                backend=backend.lower(),
                model=model,
                raw=spec,
            )

        # Try CLI model pattern: [cli]transport:model
        if match := cls._CLI_MODEL_PATTERN.match(spec):
            transport, model = match.groups()
            return cls(
                type="cli",
                provider=transport.lower(),
                model=model,
                raw=spec,
            )

        # Try CLI simple pattern: [cli]transport
        if match := cls._CLI_SIMPLE_PATTERN.match(spec):
            transport = match.group(1)
            return cls(
                type="cli",
                provider=transport.lower(),
                raw=spec,
            )

        # Invalid format
        raise ValueError(
            f"Invalid provider spec '{spec}'. Expected format: "
            "[api]provider/model or [cli]transport[:backend/model|:model]"
        )

    @classmethod
    def parse_flexible(cls, spec: str) -> "ProviderSpec":
        """Parse with fallback for simple provider IDs."""
        spec = spec.strip()
        if spec.startswith("["):
            return cls.parse(spec)
        return cls(type="cli", provider=spec.lower(), raw=spec)

    def validate(self) -> List[str]:
        """Validate the provider specification.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.type == "api":
            if self.provider not in self.KNOWN_API_PROVIDERS:
                errors.append(
                    f"Unknown API provider '{self.provider}'. "
                    f"Known: {sorted(self.KNOWN_API_PROVIDERS)}"
                )
            if not self.model:
                errors.append("API provider spec requires a model")
        else:  # cli
            if self.provider not in self.KNOWN_CLI_PROVIDERS:
                errors.append(
                    f"Unknown CLI provider '{self.provider}'. "
                    f"Known: {sorted(self.KNOWN_CLI_PROVIDERS)}"
                )
            if self.backend and self.backend not in self.KNOWN_BACKENDS:
                errors.append(
                    f"Unknown backend '{self.backend}'. "
                    f"Known: {sorted(self.KNOWN_BACKENDS)}"
                )

        return errors

    def __str__(self) -> str:
        """Return canonical string representation."""
        if self.type == "api":
            return f"[api]{self.provider}/{self.model}"
        elif self.backend:
            return f"[cli]{self.provider}:{self.backend}/{self.model}"
        elif self.model:
            return f"[cli]{self.provider}:{self.model}"
        else:
            return f"[cli]{self.provider}"


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
