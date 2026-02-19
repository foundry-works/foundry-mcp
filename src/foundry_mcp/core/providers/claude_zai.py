"""
Claude ZAI CLI provider implementation.

Bridges the `claude-zai` command-line interface (user's alias) to the ProviderContext
contract by handling availability checks, safe command construction, response parsing,
and token usage normalization. Restricts to read-only operations for security.

This is a separate provider from `claude` for users who have configured a `claude-zai`
alias with custom settings.
"""

from __future__ import annotations

from typing import Dict, Optional

from ._claude_base import (
    ALLOWED_TOOLS,
    ClaudeCLIProviderBase,
    RunnerProtocol,
)
from .base import (
    ProviderCapability,
    ProviderHooks,
    ProviderMetadata,
)
from .detectors import detect_provider_availability
from .registry import register_provider

DEFAULT_BINARY = "claude-zai"
CUSTOM_BINARY_ENV = "CLAUDE_ZAI_CLI_BINARY"

CLAUDE_ZAI_METADATA = ProviderMetadata(
    provider_id="claude-zai",
    display_name="Claude ZAI CLI",
    models=[],  # Model validation delegated to CLI
    default_model="opus",
    capabilities={
        ProviderCapability.TEXT,
        ProviderCapability.STREAMING,
        ProviderCapability.VISION,
        ProviderCapability.THINKING,
    },
    security_flags={"writes_allowed": False, "read_only": True},
    extra={"cli": "claude-zai", "output_format": "json", "allowed_tools": ALLOWED_TOOLS},
)


class ClaudeZAIProvider(ClaudeCLIProviderBase):
    """ProviderContext implementation backed by the Claude ZAI CLI with read-only restrictions."""

    _cli_label = "Claude ZAI CLI"

    def __init__(
        self,
        metadata: ProviderMetadata,
        hooks: ProviderHooks,
        *,
        model: Optional[str] = None,
        binary: Optional[str] = None,
        runner: Optional[RunnerProtocol] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ):
        super().__init__(
            metadata,
            hooks,
            model=model,
            binary=binary,
            runner=runner,
            env=env,
            timeout=timeout,
            default_binary=DEFAULT_BINARY,
            custom_binary_env=CUSTOM_BINARY_ENV,
        )


def is_claude_zai_available() -> bool:
    """Claude ZAI CLI availability check."""
    return detect_provider_availability("claude-zai")


def create_provider(
    *,
    hooks: ProviderHooks,
    model: Optional[str] = None,
    dependencies: Optional[Dict[str, object]] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> ClaudeZAIProvider:
    """
    Factory used by the provider registry.

    dependencies/overrides allow callers (or tests) to inject runner/env/binary.
    """
    dependencies = dependencies or {}
    overrides = overrides or {}
    runner = dependencies.get("runner")
    env = dependencies.get("env")
    binary = overrides.get("binary") or dependencies.get("binary")
    timeout = overrides.get("timeout")
    selected_model = overrides.get("model") if overrides.get("model") else model

    return ClaudeZAIProvider(
        metadata=CLAUDE_ZAI_METADATA,
        hooks=hooks,
        model=selected_model,  # type: ignore[arg-type]
        binary=binary,  # type: ignore[arg-type]
        runner=runner if runner is not None else None,  # type: ignore[arg-type]
        env=env if env is not None else None,  # type: ignore[arg-type]
        timeout=timeout if timeout is not None else None,  # type: ignore[arg-type]
    )


# Register the provider immediately so consumers can resolve it by id.
register_provider(
    "claude-zai",
    factory=create_provider,
    metadata=CLAUDE_ZAI_METADATA,
    availability_check=is_claude_zai_available,
    description="Claude ZAI CLI adapter with read-only tool restrictions",
    tags=("cli", "text", "vision", "thinking", "read-only"),
    replace=True,
)


__all__ = [
    "ClaudeZAIProvider",
    "create_provider",
    "is_claude_zai_available",
    "CLAUDE_ZAI_METADATA",
]
