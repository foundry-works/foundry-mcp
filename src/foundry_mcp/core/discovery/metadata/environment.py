"""
Environment tool discovery metadata.

Defines ENVIRONMENT_TOOL_METADATA, ENVIRONMENT_FEATURE_FLAGS,
and helper functions for environment tool registration and capability negotiation.
"""

from typing import Any, Dict, Optional

from ..flags import FeatureFlagDescriptor
from ..registry import ToolRegistry, get_tool_registry
from ..types import ParameterMetadata, ParameterType, ToolMetadata

# Pre-defined metadata for environment tools
ENVIRONMENT_TOOL_METADATA: Dict[str, ToolMetadata] = {
    "foundry-verify-toolchain": ToolMetadata(
        name="foundry-verify-toolchain",
        description="Verify local CLI/toolchain availability including git, python, node, and Foundry CLI. "
        "Returns readiness status for each tool with version information.",
        parameters=[
            ParameterMetadata(
                name="tools",
                type=ParameterType.ARRAY,
                description="Specific tools to check (default: all). Valid values: git, python, node, foundry",
                required=False,
                examples=[["git", "python"], ["foundry"]],
            ),
            ParameterMetadata(
                name="verbose",
                type=ParameterType.BOOLEAN,
                description="Include detailed version and path information",
                required=False,
                default=False,
            ),
        ],
        category="environment",
        version="1.0.0",
        tags=["setup", "verification", "toolchain", "cli"],
        related_tools=["foundry-verify-environment", "foundry-init-workspace"],
        examples=[
            {
                "description": "Verify all tools",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "tools": {
                            "git": {"available": True, "version": "2.39.0"},
                            "python": {"available": True, "version": "3.11.0"},
                        }
                    },
                },
            }
        ],
    ),
    "foundry-init-workspace": ToolMetadata(
        name="foundry-init-workspace",
        description="Bootstrap working directory with specs folders, config files, and git integration. "
        "Creates specs/active, specs/pending, specs/completed, specs/archived directories.",
        parameters=[
            ParameterMetadata(
                name="path",
                type=ParameterType.STRING,
                description="Target directory (default: current working directory)",
                required=False,
            ),
            ParameterMetadata(
                name="force",
                type=ParameterType.BOOLEAN,
                description="Overwrite existing configuration if present",
                required=False,
                default=False,
            ),
            ParameterMetadata(
                name="git_integration",
                type=ParameterType.BOOLEAN,
                description="Enable git hooks and integration",
                required=False,
                default=True,
            ),
        ],
        category="environment",
        version="1.0.0",
        tags=["setup", "initialization", "workspace", "config"],
        related_tools=["foundry-verify-toolchain", "foundry-detect-topology"],
        examples=[
            {
                "description": "Initialize workspace in current directory",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "created_dirs": [
                            "specs/active",
                            "specs/pending",
                            "specs/completed",
                        ],
                        "git_integration": True,
                    },
                },
            }
        ],
    ),
    "foundry-detect-topology": ToolMetadata(
        name="foundry-detect-topology",
        description="Auto-detect repository layout for specs and documentation directories. "
        "Scans directory structure to identify existing Foundry configuration.",
        parameters=[
            ParameterMetadata(
                name="path",
                type=ParameterType.STRING,
                description="Root directory to scan (default: current working directory)",
                required=False,
            ),
            ParameterMetadata(
                name="depth",
                type=ParameterType.INTEGER,
                description="Maximum directory depth to scan",
                required=False,
                default=3,
                constraints={"minimum": 1, "maximum": 10},
            ),
        ],
        category="environment",
        version="1.0.0",
        tags=["detection", "topology", "repository", "layout"],
        related_tools=["foundry-init-workspace", "foundry-verify-environment"],
        examples=[
            {
                "description": "Detect repository topology",
                "input": {"depth": 2},
                "output": {
                    "success": True,
                    "data": {
                        "specs_dir": "specs",
                        "docs_dir": "docs",
                        "has_git": True,
                        "layout_type": "standard",
                    },
                },
            }
        ],
    ),
    "foundry-verify-environment": ToolMetadata(
        name="foundry-verify-environment",
        description="Validate OS packages, runtime versions, and credential availability. "
        "Performs comprehensive environment checks beyond basic toolchain verification.",
        parameters=[
            ParameterMetadata(
                name="checks",
                type=ParameterType.ARRAY,
                description="Specific checks to run (default: all). Valid values: os, runtime, credentials",
                required=False,
                examples=[["os", "runtime"], ["credentials"]],
            ),
            ParameterMetadata(
                name="fix",
                type=ParameterType.BOOLEAN,
                description="Attempt to fix issues automatically (requires env_auto_fix feature flag)",
                required=False,
                default=False,
            ),
        ],
        category="environment",
        version="1.0.0",
        tags=["verification", "environment", "runtime", "credentials"],
        related_tools=["foundry-verify-toolchain", "foundry-detect-topology"],
        examples=[
            {
                "description": "Run all environment checks",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "os": {"platform": "darwin", "version": "14.0"},
                        "runtime": {"python": "3.11.0", "node": "20.0.0"},
                        "issues": [],
                    },
                },
            }
        ],
    ),
}


# Environment feature flags
ENVIRONMENT_FEATURE_FLAGS: Dict[str, FeatureFlagDescriptor] = {
    "environment_tools": FeatureFlagDescriptor(
        name="environment_tools",
        description="Environment setup and verification tools for Foundry workflows",
        state="beta",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=[],
    ),
    "env_auto_fix": FeatureFlagDescriptor(
        name="env_auto_fix",
        description="Automatic fix capability for environment verification issues",
        state="experimental",
        default_enabled=False,
        percentage_rollout=0,
        dependencies=["environment_tools"],
    ),
}


def register_environment_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """
    Register all environment tools in the registry.

    Args:
        registry: Optional registry to use. If None, uses global registry.

    Returns:
        The registry with environment tools registered.
    """
    if registry is None:
        registry = get_tool_registry()

    for tool_metadata in ENVIRONMENT_TOOL_METADATA.values():
        try:
            registry.register(tool_metadata)
        except ValueError:
            # Tool already registered, skip
            pass

    return registry


def get_environment_capabilities() -> Dict[str, Any]:
    """
    Get environment-related capabilities for capability negotiation.

    Returns:
        Dict with environment tool availability and feature flags.
    """
    return {
        "environment_readiness": {
            "supported": True,
            "tools": list(ENVIRONMENT_TOOL_METADATA.keys()),
            "description": "Environment verification and workspace initialization tools",
        },
        "feature_flags": {
            name: flag.to_dict() for name, flag in ENVIRONMENT_FEATURE_FLAGS.items()
        },
    }


def is_environment_tool(tool_name: str) -> bool:
    """
    Check if a tool name is an environment tool.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if tool is an environment tool
    """
    return tool_name in ENVIRONMENT_TOOL_METADATA


def get_environment_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """
    Get metadata for a specific environment tool.

    Args:
        tool_name: Name of the environment tool

    Returns:
        ToolMetadata if found, None otherwise
    """
    return ENVIRONMENT_TOOL_METADATA.get(tool_name)
