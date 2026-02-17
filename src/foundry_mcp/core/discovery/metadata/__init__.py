"""
Tool metadata sub-package for discovery.

Re-exports all tool metadata dicts, feature flags, and helper functions
from the environment, LLM, and provider metadata modules.
"""

from .environment import (  # noqa: F401
    ENVIRONMENT_FEATURE_FLAGS,
    ENVIRONMENT_TOOL_METADATA,
    get_environment_capabilities,
    get_environment_tool_metadata,
    is_environment_tool,
    register_environment_tools,
)

from .llm import (  # noqa: F401
    LLM_FEATURE_FLAGS,
    LLM_TOOL_METADATA,
    get_llm_capabilities,
    get_llm_tool_metadata,
    is_llm_tool,
    register_llm_tools,
)

from .provider import (  # noqa: F401
    PROVIDER_FEATURE_FLAGS,
    PROVIDER_TOOL_METADATA,
    get_provider_capabilities,
    get_provider_tool_metadata,
    is_provider_tool,
    register_provider_tools_discovery,
)
