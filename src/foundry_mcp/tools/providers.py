"""
Provider tools for foundry-mcp.

Exposes LLM provider management capabilities through MCP tools:
- Provider listing and status
- Provider execution
- Capability introspection

These tools bridge the provider abstraction layer to MCP clients,
enabling agents to discover, check, and invoke AI providers.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool

logger = logging.getLogger(__name__)


def register_provider_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register provider tools with the FastMCP server.

    Exposes MCP tools for:
    - provider-list: List all registered providers with availability status
    - provider-status: Get detailed status for a specific provider
    - provider-execute: Execute a prompt through a provider

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    # Import provider dependencies inside the function to avoid
    # circular imports and allow lazy loading
    from foundry_mcp.core.providers import (
        available_providers,
        describe_providers,
        check_provider_available,
        get_provider_metadata,
        get_provider_statuses,
        resolve_provider,
        ProviderHooks,
        ProviderRequest,
        ProviderCapability,
        ProviderUnavailableError,
        ProviderExecutionError,
        ProviderTimeoutError,
    )

    @canonical_tool(
        mcp,
        canonical_name="provider-list",
    )
    def provider_list(
        include_unavailable: bool = False,
    ) -> dict:
        """
        List all registered LLM providers with availability status.

        Returns registered providers sorted by priority, with availability
        status indicating which can currently be used. Useful for discovering
        what AI backends are configured.

        WHEN TO USE:
        - Discovering available AI providers
        - Checking which providers are configured
        - Before selecting a provider for execution

        Args:
            include_unavailable: Include providers that fail availability check
                                 (default: False)

        Returns:
            JSON object with:
            - providers: List of provider summaries (id, description, priority,
                        tags, available)
            - available_count: Number of available providers
            - total_count: Total registered providers
        """
        try:
            # Get full provider descriptions
            all_providers = describe_providers()

            # Filter if needed
            if not include_unavailable:
                providers = [p for p in all_providers if p.get("available", False)]
            else:
                providers = all_providers

            # Count statistics
            available_count = sum(
                1 for p in all_providers if p.get("available", False)
            )
            total_count = len(all_providers)

            return asdict(
                success_response(
                    data={
                        "providers": providers,
                        "available_count": available_count,
                        "total_count": total_count,
                    }
                )
            )

        except Exception as e:
            logger.exception("Error listing providers")
            return asdict(error_response(sanitize_error_message(e, context="providers")))

    @canonical_tool(
        mcp,
        canonical_name="provider-status",
    )
    def provider_status(
        provider_id: str,
    ) -> dict:
        """
        Get detailed status for a specific provider.

        Returns availability, metadata, capabilities, and health status
        for a registered provider. Use this to check if a provider is
        ready for use and what features it supports.

        WHEN TO USE:
        - Checking if a specific provider is available
        - Understanding provider capabilities before use
        - Debugging provider configuration issues

        Args:
            provider_id: Provider identifier (e.g., "gemini", "codex",
                        "cursor-agent", "claude", "opencode")

        Returns:
            JSON object with:
            - provider_id: The queried provider ID
            - available: Whether provider can be used
            - metadata: Provider metadata (models, capabilities, docs)
            - capabilities: List of supported capability flags
            - health: Current health status from detector
        """
        try:
            if not provider_id:
                return asdict(
                    error_response(
                        "provider_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a provider_id (e.g., 'gemini', 'codex')",
                    )
                )

            # Check if provider is available
            is_available = check_provider_available(provider_id)

            # Get metadata if available
            metadata = get_provider_metadata(provider_id)
            metadata_dict = None
            capabilities_list = None

            if metadata:
                metadata_dict = {
                    "name": metadata.display_name or metadata.provider_id,
                    "version": metadata.extra.get("version") if metadata.extra else None,
                    "default_model": metadata.default_model,
                    "supported_models": [
                        {
                            "id": m.id,
                            "name": m.display_name or m.id,
                            "context_window": m.routing_hints.get("context_window"),
                            "is_default": m.id == metadata.default_model,
                        }
                        for m in (metadata.models or [])
                    ],
                    "documentation_url": metadata.extra.get("documentation_url") if metadata.extra else None,
                    "tags": metadata.extra.get("tags", []) if metadata.extra else [],
                }
                capabilities_list = [
                    cap.value for cap in (metadata.capabilities or [])
                ]

            # Get health status from detector
            # Note: get_provider_statuses returns Dict[str, bool] (availability map)
            statuses = get_provider_statuses()
            health_status = statuses.get(provider_id)
            health_dict = None
            if health_status is not None:
                health_dict = {
                    "status": "available" if health_status else "unavailable",
                    "available": health_status,
                }

            if not is_available and metadata is None and health_dict is None:
                return asdict(
                    error_response(
                        f"Provider '{provider_id}' not found",
                        error_code="NOT_FOUND",
                        error_type="not_found",
                        remediation="Use provider-list to see available providers",
                    )
                )

            return asdict(
                success_response(
                    data={
                        "provider_id": provider_id,
                        "available": is_available,
                        "metadata": metadata_dict,
                        "capabilities": capabilities_list,
                        "health": health_dict,
                    }
                )
            )

        except Exception as e:
            logger.exception(f"Error getting provider status for {provider_id}")
            return asdict(error_response(sanitize_error_message(e, context="providers")))

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

        Sends a prompt to the specified provider and returns the result.
        Supports provider-specific model selection and generation parameters.

        WHEN TO USE:
        - Invoking an AI provider for generation
        - Executing prompts through external CLI tools
        - Multi-model consultation workflows

        LIMITATIONS:
        - Requires provider to be available and configured
        - Streaming not supported through this tool (returns complete result)
        - Some providers may have rate limits or require API keys

        Args:
            provider_id: Provider identifier (e.g., "gemini", "codex")
            prompt: The prompt text to send to the provider
            model: Optional model override (uses provider default if not specified)
            max_tokens: Maximum tokens in response (provider-specific default)
            temperature: Sampling temperature 0.0-2.0 (provider-specific default)
            timeout: Request timeout in seconds (default: 300)

        Returns:
            JSON object with:
            - provider_id: Provider that handled the request
            - model: Model used for generation
            - content: Generated text response
            - token_usage: Token counts (prompt, completion, total) if available
            - finish_reason: Why generation stopped (if available)
        """
        try:
            # Validate required parameters
            if not provider_id:
                return asdict(
                    error_response(
                        "provider_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a provider_id (e.g., 'gemini', 'codex')",
                    )
                )

            if not prompt or not prompt.strip():
                return asdict(
                    error_response(
                        "prompt is required and cannot be empty",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a non-empty prompt string",
                    )
                )

            # Check availability before attempting resolution
            if not check_provider_available(provider_id):
                return asdict(
                    error_response(
                        f"Provider '{provider_id}' is not available",
                        error_code="UNAVAILABLE",
                        error_type="unavailable",
                        remediation=(
                            "Check provider configuration and availability. "
                            "Use provider-list to see available providers."
                        ),
                    )
                )

            # Create hooks (no streaming for MCP tool invocation)
            hooks = ProviderHooks()

            # Resolve provider instance
            provider = resolve_provider(
                provider_id,
                hooks=hooks,
                model=model,
            )

            # Build request
            request = ProviderRequest(
                prompt=prompt.strip(),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout or 300,
                stream=False,  # MCP tools don't support streaming
            )

            # Execute
            result = provider.generate(request)

            # Build response
            response_data = {
                "provider_id": provider_id,
                "model": result.model_used or model or "default",
                "content": result.content,
                "finish_reason": result.status.value if result.status else None,
            }

            # Include token usage if available
            if result.tokens and result.tokens.total_tokens > 0:
                response_data["token_usage"] = {
                    "prompt_tokens": result.tokens.input_tokens,
                    "completion_tokens": result.tokens.output_tokens,
                    "total_tokens": result.tokens.total_tokens,
                }

            return asdict(success_response(data=response_data))

        except ProviderUnavailableError as e:
            logger.warning(f"Provider unavailable: {e}")
            return asdict(
                error_response(
                    f"Provider {e.provider or provider_id} is unavailable",
                    error_code="UNAVAILABLE",
                    error_type="unavailable",
                    data={"provider": e.provider} if e.provider else None,
                    remediation="Check provider configuration and availability",
                )
            )

        except ProviderTimeoutError as e:
            logger.warning(f"Provider timeout: {e}")
            return asdict(
                error_response(
                    f"Provider {e.provider or provider_id} timed out",
                    error_code="TIMEOUT",
                    error_type="unavailable",
                    data={"provider": e.provider} if e.provider else None,
                    remediation="Try again with a shorter prompt or increased timeout",
                )
            )

        except ProviderExecutionError as e:
            logger.warning(f"Provider execution error: {e}")
            return asdict(
                error_response(
                    f"Provider {e.provider or provider_id} execution failed",
                    error_code="EXECUTION_ERROR",
                    error_type="internal",
                    data={"provider": e.provider} if e.provider else None,
                    remediation="Check provider logs and try again",
                )
            )

        except Exception as e:
            logger.exception(f"Error executing provider {provider_id}")
            return asdict(error_response(sanitize_error_message(e, context="providers")))

    logger.debug("Provider tools registered")
