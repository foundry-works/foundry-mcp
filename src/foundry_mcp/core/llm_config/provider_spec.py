"""Provider specification (unified priority notation) for CLI-based providers."""

import re
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class ProviderSpec:
    """Parsed provider specification from bracket notation.

    Supports bracket-prefix notation for CLI provider configuration:
        - [cli]gemini:pro               -> CLI provider with model
        - [cli]claude:opus              -> CLI provider with model
        - [cli]opencode:openai/gpt-5.2  -> CLI provider routing to backend
        - [cli]codex                    -> CLI provider with default model

    Grammar:
        spec       := "[cli]" cli_spec
        cli_spec   := transport (":" backend "/" model | ":" model | "")

    Attributes:
        type: Provider type - always "cli" for CLI tools
        provider: Provider/transport identifier (gemini, opencode, etc.)
        backend: Optional backend for CLI routing (openai, anthropic, gemini)
        model: Optional model identifier (pro, opus, etc.)
        raw: Original specification string for error messages
    """

    type: Literal["cli"]
    provider: str
    backend: Optional[str] = None
    model: Optional[str] = None
    raw: str = ""

    # Known providers for validation
    KNOWN_CLI_PROVIDERS = {"gemini", "codex", "cursor-agent", "opencode", "claude"}
    KNOWN_BACKENDS = {"openai", "anthropic", "gemini", "local"}

    # Regex patterns for parsing
    _CLI_FULL_PATTERN = re.compile(r"^\[cli\]([^:]+):([^/]+)/(.+)$")  # transport:backend/model
    _CLI_MODEL_PATTERN = re.compile(r"^\[cli\]([^:]+):([^/]+)$")  # transport:model
    _CLI_SIMPLE_PATTERN = re.compile(r"^\[cli\]([^:]+)$")  # transport only

    @classmethod
    def parse(cls, spec: str) -> "ProviderSpec":
        """Parse a provider specification string.

        Args:
            spec: Provider spec in bracket notation (e.g., "[cli]claude:opus")

        Returns:
            ProviderSpec instance with parsed components

        Raises:
            ValueError: If the spec format is invalid or uses the removed [api] prefix

        Examples:
            >>> ProviderSpec.parse("[cli]gemini:pro")
            ProviderSpec(type='cli', provider='gemini', model='pro')

            >>> ProviderSpec.parse("[cli]opencode:openai/gpt-5.2")
            ProviderSpec(type='cli', provider='opencode', backend='openai', model='gpt-5.2')
        """
        spec = spec.strip()

        if not spec:
            raise ValueError("Provider spec cannot be empty")

        # Reject removed [api] specs
        if spec.startswith("[api]"):
            raise ValueError(
                f"API provider specs are no longer supported: '{spec}'. "
                "Use CLI providers instead (e.g., [cli]claude:opus)."
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
        raise ValueError(f"Invalid provider spec '{spec}'. Expected format: [cli]transport[:backend/model|:model]")

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

        if self.type != "cli":
            errors.append(f"Unsupported provider type '{self.type}'. Only 'cli' is supported.")
            return errors

        if self.provider not in self.KNOWN_CLI_PROVIDERS:
            errors.append(f"Unknown CLI provider '{self.provider}'. Known: {sorted(self.KNOWN_CLI_PROVIDERS)}")
        if self.backend and self.backend not in self.KNOWN_BACKENDS:
            errors.append(f"Unknown backend '{self.backend}'. Known: {sorted(self.KNOWN_BACKENDS)}")

        return errors

    def __str__(self) -> str:
        """Return canonical string representation."""
        if self.backend:
            return f"[cli]{self.provider}:{self.backend}/{self.model}"
        elif self.model:
            return f"[cli]{self.provider}:{self.model}"
        else:
            return f"[cli]{self.provider}"
