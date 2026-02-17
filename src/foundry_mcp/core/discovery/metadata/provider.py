"""
Provider tool discovery metadata.

Defines PROVIDER_TOOL_METADATA, PROVIDER_FEATURE_FLAGS,
and helper functions for provider tool registration and capability negotiation.
"""

from typing import Any, Dict, Optional

from ..flags import FeatureFlagDescriptor
from ..registry import ToolRegistry, get_tool_registry
from ..types import ParameterMetadata, ParameterType, ToolMetadata

# Pre-defined metadata for provider tools
PROVIDER_TOOL_METADATA: Dict[str, ToolMetadata] = {
    "provider-list": ToolMetadata(
        name="provider-list",
        description="List all registered LLM providers with availability status. "
        "Returns providers sorted by priority, with availability indicating "
        "which can currently be used for execution.",
        parameters=[
            ParameterMetadata(
                name="include_unavailable",
                type=ParameterType.BOOLEAN,
                description="Include providers that fail availability check",
                required=False,
                default=False,
            ),
        ],
        category="providers",
        version="1.0.0",
        tags=["providers", "discovery", "status", "availability"],
        related_tools=["provider-status", "provider-execute"],
        examples=[
            {
                "description": "List available providers",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "providers": [
                            {
                                "id": "gemini",
                                "description": "Google Gemini CLI provider",
                                "priority": 10,
                                "tags": ["cli", "external"],
                                "available": True,
                            },
                            {
                                "id": "codex",
                                "description": "OpenAI Codex CLI provider",
                                "priority": 5,
                                "tags": ["cli", "external"],
                                "available": True,
                            },
                        ],
                        "available_count": 2,
                        "total_count": 5,
                    },
                },
            },
            {
                "description": "Include unavailable providers",
                "input": {"include_unavailable": True},
                "output": {
                    "success": True,
                    "data": {
                        "providers": [
                            {"id": "gemini", "available": True},
                            {"id": "codex", "available": True},
                            {"id": "opencode", "available": False},
                        ],
                        "available_count": 2,
                        "total_count": 3,
                    },
                },
            },
        ],
    ),
    "provider-status": ToolMetadata(
        name="provider-status",
        description="Get detailed status for a specific LLM provider. "
        "Returns availability, metadata, capabilities, and health status "
        "for debugging and capability introspection.",
        parameters=[
            ParameterMetadata(
                name="provider_id",
                type=ParameterType.STRING,
                description="Provider identifier (e.g., 'gemini', 'codex', 'cursor-agent')",
                required=True,
                examples=["gemini", "codex", "cursor-agent", "claude", "opencode"],
            ),
        ],
        category="providers",
        version="1.0.0",
        tags=["providers", "status", "health", "capabilities"],
        related_tools=["provider-list", "provider-execute"],
        examples=[
            {
                "description": "Get status for Gemini provider",
                "input": {"provider_id": "gemini"},
                "output": {
                    "success": True,
                    "data": {
                        "provider_id": "gemini",
                        "available": True,
                        "metadata": {
                            "name": "Gemini",
                            "version": "1.0.0",
                            "default_model": "gemini-pro",
                            "supported_models": [
                                {"id": "gemini-pro", "name": "Gemini Pro", "is_default": True}
                            ],
                        },
                        "capabilities": ["text_generation", "streaming"],
                        "health": {"status": "healthy", "reason": None},
                    },
                },
            },
            {
                "description": "Provider not found",
                "input": {"provider_id": "unknown"},
                "output": {
                    "success": False,
                    "error": "Provider 'unknown' not found",
                    "data": {"error_code": "NOT_FOUND"},
                },
            },
        ],
    ),
    "provider-execute": ToolMetadata(
        name="provider-execute",
        description="Execute a prompt through a specified LLM provider. "
        "Sends a prompt to the provider and returns the complete result. "
        "Supports model selection and generation parameters.",
        parameters=[
            ParameterMetadata(
                name="provider_id",
                type=ParameterType.STRING,
                description="Provider identifier (e.g., 'gemini', 'codex')",
                required=True,
                examples=["gemini", "codex", "cursor-agent"],
            ),
            ParameterMetadata(
                name="prompt",
                type=ParameterType.STRING,
                description="The prompt text to send to the provider",
                required=True,
                examples=["Explain the concept of dependency injection"],
            ),
            ParameterMetadata(
                name="model",
                type=ParameterType.STRING,
                description="Model override (uses provider default if not specified)",
                required=False,
                examples=["gemini-pro", "gpt-4o"],
            ),
            ParameterMetadata(
                name="max_tokens",
                type=ParameterType.INTEGER,
                description="Maximum tokens in response",
                required=False,
                constraints={"minimum": 1, "maximum": 100000},
            ),
            ParameterMetadata(
                name="temperature",
                type=ParameterType.NUMBER,
                description="Sampling temperature (0.0-2.0)",
                required=False,
                constraints={"minimum": 0.0, "maximum": 2.0},
            ),
            ParameterMetadata(
                name="timeout",
                type=ParameterType.INTEGER,
                description="Request timeout in seconds",
                required=False,
                default=300,
                constraints={"minimum": 1, "maximum": 3600},
            ),
        ],
        category="providers",
        version="1.0.0",
        tags=["providers", "execution", "generation", "llm"],
        related_tools=["provider-list", "provider-status"],
        rate_limit="60/minute",
        examples=[
            {
                "description": "Execute prompt through Gemini",
                "input": {
                    "provider_id": "gemini",
                    "prompt": "What is dependency injection?",
                },
                "output": {
                    "success": True,
                    "data": {
                        "provider_id": "gemini",
                        "model": "gemini-pro",
                        "content": "Dependency injection is a design pattern...",
                        "token_usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 150,
                            "total_tokens": 160,
                        },
                        "finish_reason": "stop",
                    },
                },
            },
            {
                "description": "Provider unavailable",
                "input": {"provider_id": "opencode", "prompt": "Hello"},
                "output": {
                    "success": False,
                    "error": "Provider 'opencode' is not available",
                    "data": {"error_code": "UNAVAILABLE"},
                },
            },
        ],
    ),
}


# Provider feature flags for capability negotiation
PROVIDER_FEATURE_FLAGS: Dict[str, FeatureFlagDescriptor] = {
    "provider_tools": FeatureFlagDescriptor(
        name="provider_tools",
        description="MCP tools for LLM provider management and execution",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=[],
    ),
    "provider_multi_model": FeatureFlagDescriptor(
        name="provider_multi_model",
        description="Support for multiple models per provider",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["provider_tools"],
    ),
    "provider_streaming": FeatureFlagDescriptor(
        name="provider_streaming",
        description="Streaming response support for providers (not exposed via MCP tools)",
        state="beta",
        default_enabled=False,
        percentage_rollout=0,
        dependencies=["provider_tools"],
    ),
    "provider_rate_limiting": FeatureFlagDescriptor(
        name="provider_rate_limiting",
        description="Rate limiting and circuit breaker support for providers",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["provider_tools"],
    ),
}


def register_provider_tools_discovery(
    registry: Optional[ToolRegistry] = None,
) -> ToolRegistry:
    """
    Register all provider tools in the discovery registry.

    Args:
        registry: Optional registry to use. If None, uses global registry.

    Returns:
        The registry with provider tools registered.
    """
    if registry is None:
        registry = get_tool_registry()

    for tool_metadata in PROVIDER_TOOL_METADATA.values():
        try:
            registry.register(tool_metadata)
        except ValueError:
            # Tool already registered, skip
            pass

    return registry


def get_provider_capabilities() -> Dict[str, Any]:
    """
    Get provider-related capabilities for capability negotiation.

    Returns:
        Dict with provider tool availability and feature flags.
    """
    return {
        "provider_tools": {
            "supported": True,
            "tools": list(PROVIDER_TOOL_METADATA.keys()),
            "description": "LLM provider management, status, and execution tools",
        },
        "supported_providers": {
            "built_in": ["gemini", "codex", "cursor-agent", "claude", "opencode"],
            "extensible": True,
            "description": "Pluggable provider architecture with registry support",
        },
        "feature_flags": {
            name: flag.to_dict() for name, flag in PROVIDER_FEATURE_FLAGS.items()
        },
    }


def is_provider_tool(tool_name: str) -> bool:
    """
    Check if a tool name is a provider tool.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if tool is a provider tool
    """
    return tool_name in PROVIDER_TOOL_METADATA


def get_provider_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """
    Get metadata for a specific provider tool.

    Args:
        tool_name: Name of the provider tool

    Returns:
        ToolMetadata if found, None otherwise
    """
    return PROVIDER_TOOL_METADATA.get(tool_name)
