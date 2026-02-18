"""
Tool metadata and discovery module for foundry-mcp.

Provides dataclasses and utilities for tool registration, discovery,
capability negotiation, and deprecation handling per MCP best practices
(dev_docs/mcp_best_practices/13-tool-discovery.md).

All public symbols are re-exported from sub-modules for backward compatibility.
"""

# --- Types (leaf module) ---

from .types import (  # noqa: F401
    SCHEMA_VERSION,
    ParameterMetadata,
    ParameterType,
    ToolMetadata,
)

# --- Registry ---

from .registry import (  # noqa: F401
    ToolRegistry,
    get_tool_registry,
)

# --- Capabilities ---

from .capabilities import (  # noqa: F401
    ServerCapabilities,
    get_capabilities,
    negotiate_capabilities,
    set_capabilities,
)

# --- Deprecation ---

from .deprecation import (  # noqa: F401
    deprecated_tool,
    get_deprecation_info,
    is_deprecated,
)

# --- Feature flag descriptors (used by metadata modules) ---

from .flags import (  # noqa: F401
    FeatureFlagDescriptor,
)

# --- Metadata (environment, LLM, provider) ---

from .metadata import (  # noqa: F401
    ENVIRONMENT_FEATURE_FLAGS,
    ENVIRONMENT_TOOL_METADATA,
    LLM_FEATURE_FLAGS,
    LLM_TOOL_METADATA,
    PROVIDER_FEATURE_FLAGS,
    PROVIDER_TOOL_METADATA,
    get_environment_capabilities,
    get_environment_tool_metadata,
    get_llm_capabilities,
    get_llm_tool_metadata,
    get_provider_capabilities,
    get_provider_tool_metadata,
    is_environment_tool,
    is_llm_tool,
    is_provider_tool,
    register_environment_tools,
    register_llm_tools,
    register_provider_tools_discovery,
)
