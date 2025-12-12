"""
Provider tools for foundry-mcp.

These wrappers preserve the legacy provider-* tool names while delegating to the
unified provider(action=...) router.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.provider import legacy_provider_action


def register_provider_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register provider tools with the FastMCP server."""

    @canonical_tool(
        mcp,
        canonical_name="provider-list",
    )
    def provider_list(
        include_unavailable: bool = False,
    ) -> dict:
        """
        List all registered LLM providers with availability status.

        Args:
            include_unavailable: Include providers that fail availability check
                                 (default: False)
        """

        return legacy_provider_action(
            "list",
            config=config,
            include_unavailable=include_unavailable,
        )

    @canonical_tool(
        mcp,
        canonical_name="provider-status",
    )
    def provider_status(
        provider_id: str,
    ) -> dict:
        """
        Get detailed status for a specific provider.

        Args:
            provider_id: Provider identifier (e.g., "gemini", "codex")
        """

        return legacy_provider_action(
            "status",
            config=config,
            provider_id=provider_id,
        )

    @canonical_tool(
        mcp,
        canonical_name="provider-execute",
    )
    def provider_execute(
        provider_id: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        Execute a prompt through a specified LLM provider.

        Args:
            provider_id: Provider identifier
            prompt: Prompt text to send to the provider
            model: Optional model override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature 0.0-2.0
            timeout: Request timeout in seconds (default 300)
        """

        return legacy_provider_action(
            "execute",
            config=config,
            provider_id=provider_id,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
