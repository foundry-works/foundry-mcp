"""
Tool metadata and discovery module for foundry-mcp.

Provides dataclasses and utilities for tool registration, discovery,
capability negotiation, and deprecation handling per MCP best practices
(docs/mcp_best_practices/13-tool-discovery.md).
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Schema version for discovery responses
SCHEMA_VERSION = "1.0.0"


class ParameterType(str, Enum):
    """Types for tool parameters in JSON Schema."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterMetadata:
    """
    Metadata for a tool parameter.

    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, boolean, etc.)
        description: Human-readable description
        required: Whether parameter is required
        default: Default value if not provided
        constraints: JSON Schema constraints (pattern, min, max, enum, etc.)
        examples: Example values for documentation
    """

    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)


@dataclass
class ToolMetadata:
    """
    Comprehensive metadata for an MCP tool.

    Used for tool registration, discovery, and documentation generation.
    Supports categorization, versioning, deprecation, and rate limiting.

    Attributes:
        name: Unique tool identifier
        description: Human-readable description (supports markdown)
        parameters: List of parameter metadata
        category: Tool category for grouping (e.g., "files", "users", "code")
        version: Semantic version of the tool interface
        deprecated: Whether tool is deprecated
        deprecation_message: Message explaining deprecation and migration path
        rate_limit: Rate limit description (e.g., "100/minute")
        examples: List of usage examples with input/output
        related_tools: Names of related tools
        tags: Semantic tags for filtering
    """

    name: str
    description: str
    parameters: List[ParameterMetadata] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    rate_limit: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate JSON Schema from tool metadata.

        Returns:
            JSON Schema dict for tool input validation
        """
        type_mapping = {
            ParameterType.STRING: "string",
            ParameterType.INTEGER: "integer",
            ParameterType.NUMBER: "number",
            ParameterType.BOOLEAN: "boolean",
            ParameterType.ARRAY: "array",
            ParameterType.OBJECT: "object",
        }

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": type_mapping[param.type],
                "description": param.description,
            }

            if param.default is not None:
                prop["default"] = param.default

            if param.examples:
                prop["examples"] = param.examples

            # Add constraints
            prop.update(param.constraints)

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def to_summary(self) -> Dict[str, Any]:
        """
        Generate summary for tool listing.

        Returns:
            Compact dict for list_tools responses
        """
        desc = self.description
        if len(desc) > 200:
            desc = desc[:197] + "..."

        return {
            "name": self.name,
            "description": desc,
            "category": self.category,
            "version": self.version,
            "deprecated": self.deprecated,
            "tags": self.tags,
        }

    def to_detailed(self) -> Dict[str, Any]:
        """
        Generate detailed info for get_tool_schema responses.

        Returns:
            Full dict including schema, examples, and related tools
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "schema": self.to_json_schema(),
            "examples": self.examples,
            "related_tools": self.related_tools,
            "tags": self.tags,
            "deprecated": self.deprecated,
        }

        if self.rate_limit:
            result["rate_limit"] = self.rate_limit

        if self.deprecation_message:
            result["deprecation_message"] = self.deprecation_message

        return result


class ToolRegistry:
    """
    Central registry for MCP tool metadata.

    Provides tool registration, discovery, and filtering capabilities.
    Used by MCP servers to expose tool information to clients.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ToolMetadata(
        ...     name="get_user",
        ...     description="Get user by ID",
        ...     category="users",
        ... ))
        >>> tools = registry.list_tools(category="users")
    """

    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: ToolMetadata) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool metadata to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

        # Update category index
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False

        tool = self._tools.pop(name)
        if tool.category in self._categories:
            self._categories[tool.category].remove(name)
            if not self._categories[tool.category]:
                del self._categories[tool.category]

        return True

    def get(self, name: str) -> Optional[ToolMetadata]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List available tools with filtering.

        Args:
            category: Filter by category
            tag: Filter by tag
            include_deprecated: Include deprecated tools (default: False)

        Returns:
            List of tool summaries
        """
        tools = list(self._tools.values())

        # Apply filters
        if category:
            tools = [t for t in tools if t.category == category]

        if tag:
            tools = [t for t in tools if tag in t.tags]

        if not include_deprecated:
            tools = [t for t in tools if not t.deprecated]

        return [t.to_summary() for t in tools]

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed schema for a specific tool.

        Args:
            name: Tool name

        Returns:
            Detailed tool info including schema, or None if not found
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.to_detailed()

    def list_categories(self) -> List[Dict[str, Any]]:
        """
        List tool categories with descriptions.

        Returns:
            List of categories with tool counts
        """
        result = []
        for category, tool_names in sorted(self._categories.items()):
            # Filter out deprecated tools from count
            active_count = sum(
                1
                for name in tool_names
                if name in self._tools and not self._tools[name].deprecated
            )

            result.append(
                {
                    "name": category,
                    "tool_count": active_count,
                    "tools": tool_names,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with total counts and breakdown
        """
        total = len(self._tools)
        deprecated = sum(1 for t in self._tools.values() if t.deprecated)

        return {
            "total_tools": total,
            "active_tools": total - deprecated,
            "deprecated_tools": deprecated,
            "categories": len(self._categories),
        }


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


@dataclass
class ServerCapabilities:
    """
    Server capabilities for client negotiation.

    Clients call get_capabilities() to understand what features are supported
    before making assumptions about available functionality.

    Attributes:
        response_version: Response contract version (e.g., "response-v2")
        supports_streaming: Whether server supports streaming responses
        supports_batch: Whether server supports batch operations
        supports_pagination: Whether server supports cursor-based pagination
        max_batch_size: Maximum items in a batch request
        rate_limit_headers: Whether responses include rate limit headers
        supported_formats: List of supported response formats
        feature_flags_enabled: Whether feature flags are active
    """

    response_version: str = "response-v2"
    supports_streaming: bool = False
    supports_batch: bool = True
    supports_pagination: bool = True
    max_batch_size: int = 100
    rate_limit_headers: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ["json"])
    feature_flags_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert capabilities to dict for response.

        Returns:
            Dict suitable for capability negotiation responses
        """
        return {
            "response_version": self.response_version,
            "streaming": self.supports_streaming,
            "batch_operations": self.supports_batch,
            "pagination": self.supports_pagination,
            "max_batch_size": self.max_batch_size,
            "rate_limit_headers": self.rate_limit_headers,
            "formats": self.supported_formats,
            "feature_flags": self.feature_flags_enabled,
        }


# Global capabilities instance
_capabilities: Optional[ServerCapabilities] = None


def get_capabilities() -> Dict[str, Any]:
    """
    Get server capabilities for client negotiation.

    Clients should call this to understand server features before making
    assumptions about available functionality.

    Returns:
        Dict with capabilities, server version, and API version
    """
    global _capabilities
    if _capabilities is None:
        _capabilities = ServerCapabilities()

    return {
        "schema_version": SCHEMA_VERSION,
        "capabilities": _capabilities.to_dict(),
        "server_version": "1.0.0",
        "api_version": "2024-11-01",
    }


def negotiate_capabilities(
    requested_version: Optional[str] = None,
    requested_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Negotiate capabilities with client.

    Args:
        requested_version: Desired response version
        requested_features: List of requested feature names

    Returns:
        Dict with negotiated capabilities and any warnings
    """
    global _capabilities
    if _capabilities is None:
        _capabilities = ServerCapabilities()

    negotiated: Dict[str, Any] = {}
    warnings: List[str] = []

    # Version negotiation
    if requested_version:
        if requested_version == _capabilities.response_version:
            negotiated["response_version"] = requested_version
        else:
            negotiated["response_version"] = _capabilities.response_version
            warnings.append(
                f"Requested version '{requested_version}' not supported, "
                f"using '{_capabilities.response_version}'"
            )

    # Feature negotiation
    available_features = {
        "streaming": _capabilities.supports_streaming,
        "batch": _capabilities.supports_batch,
        "pagination": _capabilities.supports_pagination,
        "rate_limit_headers": _capabilities.rate_limit_headers,
        "feature_flags": _capabilities.feature_flags_enabled,
    }

    if requested_features:
        negotiated["features"] = {}
        for feature in requested_features:
            if feature in available_features:
                negotiated["features"][feature] = available_features[feature]
            else:
                negotiated["features"][feature] = False
                warnings.append(f"Feature '{feature}' not recognized")

    return {
        "schema_version": SCHEMA_VERSION,
        "negotiated": negotiated,
        "warnings": warnings if warnings else None,
    }


def set_capabilities(capabilities: ServerCapabilities) -> None:
    """
    Set server capabilities (for testing or custom configuration).

    Args:
        capabilities: ServerCapabilities instance to use
    """
    global _capabilities
    _capabilities = capabilities


# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def deprecated_tool(
    replacement: str,
    removal_version: str,
) -> Callable[[F], F]:
    """
    Decorator to mark MCP tools as deprecated.

    Modifies the function's docstring to include deprecation notice and
    adds deprecation warning to response meta.warnings.

    Args:
        replacement: Name of the tool that replaces this one
        removal_version: Version in which this tool will be removed

    Returns:
        Decorated function that adds deprecation warnings to responses

    Example:
        >>> @mcp.tool()
        ... @deprecated_tool(replacement="get_user", removal_version="3.0.0")
        ... def fetch_user(user_id: str) -> dict:
        ...     '''Fetch user by ID (deprecated).'''
        ...     return get_user(user_id)
    """

    def decorator(func: F) -> F:
        original_doc = func.__doc__ or ""

        # Update docstring with deprecation notice
        func.__doc__ = f"""[DEPRECATED] {original_doc}

        ⚠️  This tool is deprecated and will be removed in version {removal_version}.
        Use '{replacement}' instead.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Add deprecation warning to response if it has meta
            if isinstance(result, dict):
                if "meta" not in result:
                    result["meta"] = {"version": "response-v2"}

                if "warnings" not in result["meta"]:
                    result["meta"]["warnings"] = []

                deprecation_warning = (
                    f"DEPRECATED: '{func.__name__}' will be removed in {removal_version}. "
                    f"Use '{replacement}' instead."
                )

                if deprecation_warning not in result["meta"]["warnings"]:
                    result["meta"]["warnings"].append(deprecation_warning)

            return result

        # Store deprecation metadata on the wrapper for introspection
        wrapper._deprecated = True  # type: ignore[attr-defined]
        wrapper._replacement = replacement  # type: ignore[attr-defined]
        wrapper._removal_version = removal_version  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def is_deprecated(func: Callable[..., Any]) -> bool:
    """
    Check if a function is marked as deprecated.

    Args:
        func: Function to check

    Returns:
        True if function has @deprecated_tool decorator
    """
    return getattr(func, "_deprecated", False)


def get_deprecation_info(func: Callable[..., Any]) -> Optional[Dict[str, str]]:
    """
    Get deprecation info for a deprecated function.

    Args:
        func: Function to check

    Returns:
        Dict with replacement and removal_version, or None if not deprecated
    """
    if not is_deprecated(func):
        return None

    return {
        "replacement": getattr(func, "_replacement", ""),
        "removal_version": getattr(func, "_removal_version", ""),
    }


# =============================================================================
# Environment Tools Discovery Metadata
# =============================================================================


# Pre-defined metadata for environment tools
ENVIRONMENT_TOOL_METADATA: Dict[str, ToolMetadata] = {
    "sdd-verify-toolchain": ToolMetadata(
        name="sdd-verify-toolchain",
        description="Verify local CLI/toolchain availability including git, python, node, and SDD CLI. "
        "Returns readiness status for each tool with version information.",
        parameters=[
            ParameterMetadata(
                name="tools",
                type=ParameterType.ARRAY,
                description="Specific tools to check (default: all). Valid values: git, python, node, sdd",
                required=False,
                examples=[["git", "python"], ["sdd"]],
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
        related_tools=["sdd-verify-environment", "sdd-init-workspace"],
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
    "sdd-init-workspace": ToolMetadata(
        name="sdd-init-workspace",
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
        related_tools=["sdd-verify-toolchain", "sdd-detect-topology"],
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
    "sdd-detect-topology": ToolMetadata(
        name="sdd-detect-topology",
        description="Auto-detect repository layout for specs and documentation directories. "
        "Scans directory structure to identify existing SDD configuration.",
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
        related_tools=["sdd-init-workspace", "sdd-verify-environment"],
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
    "sdd-verify-environment": ToolMetadata(
        name="sdd-verify-environment",
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
        related_tools=["sdd-verify-toolchain", "sdd-detect-topology"],
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


@dataclass
class FeatureFlagDescriptor:
    """
    Descriptor for a feature flag used in capability negotiation.

    Attributes:
        name: Unique flag identifier
        description: Human-readable description
        state: Lifecycle state (experimental, beta, stable, deprecated)
        default_enabled: Whether flag is enabled by default
        percentage_rollout: Rollout percentage (0-100)
        dependencies: List of other flags this depends on
    """

    name: str
    description: str
    state: str = "beta"
    default_enabled: bool = False
    percentage_rollout: int = 0
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state,
            "default_enabled": self.default_enabled,
            "percentage_rollout": self.percentage_rollout,
            "dependencies": self.dependencies,
        }


# Environment feature flags
ENVIRONMENT_FEATURE_FLAGS: Dict[str, FeatureFlagDescriptor] = {
    "environment_tools": FeatureFlagDescriptor(
        name="environment_tools",
        description="Environment setup and verification tools for SDD workflows",
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


# =============================================================================
# LLM-Powered Tools Discovery Metadata
# =============================================================================


# Pre-defined metadata for LLM-powered tools
LLM_TOOL_METADATA: Dict[str, ToolMetadata] = {
    "spec-review": ToolMetadata(
        name="spec-review",
        description="Run an LLM-powered review session on a specification. "
        "Performs intelligent spec analysis and generates improvement suggestions. "
        "Supports multiple review types and external AI tool integration.",
        parameters=[
            ParameterMetadata(
                name="spec_id",
                type=ParameterType.STRING,
                description="Specification ID to review",
                required=True,
                examples=["feature-auth-2025-01-15-001", "bugfix-cache-2025-02-01-001"],
            ),
            ParameterMetadata(
                name="review_type",
                type=ParameterType.STRING,
                description="Type of review to perform",
                required=False,
                default="quick",
                constraints={"enum": ["quick", "full", "security", "feasibility"]},
                examples=["quick", "full", "security"],
            ),
            ParameterMetadata(
                name="tools",
                type=ParameterType.STRING,
                description="Comma-separated list of review tools to use",
                required=False,
                examples=["cursor-agent", "gemini,codex", "cursor-agent,gemini,codex"],
            ),
            ParameterMetadata(
                name="model",
                type=ParameterType.STRING,
                description="LLM model to use for review (default: from config)",
                required=False,
                examples=["gpt-4o", "claude-3-sonnet"],
            ),
            ParameterMetadata(
                name="path",
                type=ParameterType.STRING,
                description="Project root path (default: current directory)",
                required=False,
            ),
            ParameterMetadata(
                name="dry_run",
                type=ParameterType.BOOLEAN,
                description="Show what would be reviewed without executing",
                required=False,
                default=False,
            ),
        ],
        category="llm",
        version="1.0.0",
        tags=["review", "llm", "ai", "analysis", "quality"],
        related_tools=["review-list-tools", "review-list-plan-tools", "spec-review-fidelity"],
        examples=[
            {
                "description": "Quick review of a specification",
                "input": {"spec_id": "feature-auth-001", "review_type": "quick"},
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "feature-auth-001",
                        "review_type": "quick",
                        "findings": [],
                        "suggestions": ["Consider adding error handling"],
                    },
                },
            },
            {
                "description": "Security review with multiple tools",
                "input": {
                    "spec_id": "payment-flow-001",
                    "review_type": "security",
                    "tools": "cursor-agent,gemini",
                },
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "payment-flow-001",
                        "review_type": "security",
                        "findings": [{"severity": "medium", "issue": "Missing rate limiting"}],
                    },
                },
            },
        ],
    ),
    "review-list-tools": ToolMetadata(
        name="review-list-tools",
        description="List available review tools and pipelines. "
        "Returns the set of external AI tools that can be used for spec reviews "
        "along with their availability status.",
        parameters=[],
        category="llm",
        version="1.0.0",
        tags=["review", "discovery", "tools", "configuration"],
        related_tools=["spec-review", "review-list-plan-tools"],
        examples=[
            {
                "description": "List all available review tools",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "tools": [
                            {"name": "cursor-agent", "available": True, "version": "1.0.0"},
                            {"name": "gemini", "available": True, "version": "2.0"},
                            {"name": "codex", "available": False, "version": None},
                        ],
                        "llm_status": {"configured": True, "provider": "openai"},
                    },
                },
            }
        ],
    ),
    "review-list-plan-tools": ToolMetadata(
        name="review-list-plan-tools",
        description="Enumerate review toolchains available for plan analysis. "
        "Returns tools specifically designed for reviewing SDD plans "
        "including their capabilities and recommended usage.",
        parameters=[],
        category="llm",
        version="1.0.0",
        tags=["review", "planning", "tools", "recommendations"],
        related_tools=["spec-review", "review-list-tools"],
        examples=[
            {
                "description": "List plan review toolchains",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "plan_tools": [
                            {
                                "name": "quick-review",
                                "llm_required": False,
                                "status": "available",
                            },
                            {
                                "name": "full-review",
                                "llm_required": True,
                                "status": "available",
                            },
                        ],
                        "recommendations": ["Use 'full-review' for comprehensive analysis"],
                    },
                },
            }
        ],
    ),
    "spec-review-fidelity": ToolMetadata(
        name="spec-review-fidelity",
        description="Compare implementation against specification and identify deviations. "
        "Performs a fidelity review to verify that code implementation matches "
        "the specification requirements. Uses AI consultation for comprehensive analysis.",
        parameters=[
            ParameterMetadata(
                name="spec_id",
                type=ParameterType.STRING,
                description="Specification ID to review against",
                required=True,
                examples=["feature-auth-001", "api-v2-migration-001"],
            ),
            ParameterMetadata(
                name="task_id",
                type=ParameterType.STRING,
                description="Review specific task implementation (mutually exclusive with phase_id)",
                required=False,
            ),
            ParameterMetadata(
                name="phase_id",
                type=ParameterType.STRING,
                description="Review entire phase implementation (mutually exclusive with task_id)",
                required=False,
            ),
            ParameterMetadata(
                name="files",
                type=ParameterType.ARRAY,
                description="Review specific file(s) only",
                required=False,
                examples=[["src/auth.py"], ["src/api/users.py", "src/api/auth.py"]],
            ),
            ParameterMetadata(
                name="use_ai",
                type=ParameterType.BOOLEAN,
                description="Enable AI consultation for analysis",
                required=False,
                default=True,
            ),
            ParameterMetadata(
                name="ai_tools",
                type=ParameterType.ARRAY,
                description="Specific AI tools to consult",
                required=False,
                examples=[["cursor-agent", "gemini"]],
            ),
            ParameterMetadata(
                name="consensus_threshold",
                type=ParameterType.INTEGER,
                description="Minimum models that must agree",
                required=False,
                default=2,
                constraints={"minimum": 1, "maximum": 5},
            ),
            ParameterMetadata(
                name="incremental",
                type=ParameterType.BOOLEAN,
                description="Only review changed files since last run",
                required=False,
                default=False,
            ),
        ],
        category="llm",
        version="1.0.0",
        tags=["fidelity", "review", "verification", "compliance", "llm"],
        related_tools=["spec-review"],
        rate_limit="20/hour",
        examples=[
            {
                "description": "Fidelity review for a phase",
                "input": {"spec_id": "feature-auth-001", "phase_id": "phase-1"},
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "feature-auth-001",
                        "scope": "phase",
                        "verdict": "pass",
                        "deviations": [],
                        "consensus": {"models_consulted": 3, "agreement": "unanimous"},
                    },
                },
            },
            {
                "description": "Fidelity review with deviations found",
                "input": {"spec_id": "api-v2-001", "task_id": "task-2-1"},
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "api-v2-001",
                        "scope": "task",
                        "verdict": "partial",
                        "deviations": [
                            {
                                "task_id": "task-2-1",
                                "type": "missing_implementation",
                                "severity": "high",
                            }
                        ],
                    },
                },
            },
        ],
    ),
}


# LLM feature flags for capability negotiation
LLM_FEATURE_FLAGS: Dict[str, FeatureFlagDescriptor] = {
    "llm_tools": FeatureFlagDescriptor(
        name="llm_tools",
        description="LLM-powered review and documentation tools",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=[],
    ),
    "llm_multi_provider": FeatureFlagDescriptor(
        name="llm_multi_provider",
        description="Multi-provider AI tool support (cursor-agent, gemini, codex)",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["llm_tools"],
    ),
    "llm_fidelity_review": FeatureFlagDescriptor(
        name="llm_fidelity_review",
        description="AI-powered fidelity review with consensus mechanism",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["llm_tools", "llm_multi_provider"],
    ),
    "llm_data_only_fallback": FeatureFlagDescriptor(
        name="llm_data_only_fallback",
        description="Graceful fallback to data-only responses when LLM unavailable",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["llm_tools"],
    ),
}


def register_llm_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """
    Register all LLM-powered tools in the registry.

    Args:
        registry: Optional registry to use. If None, uses global registry.

    Returns:
        The registry with LLM tools registered.
    """
    if registry is None:
        registry = get_tool_registry()

    for tool_metadata in LLM_TOOL_METADATA.values():
        try:
            registry.register(tool_metadata)
        except ValueError:
            # Tool already registered, skip
            pass

    return registry


def get_llm_capabilities() -> Dict[str, Any]:
    """
    Get LLM-related capabilities for capability negotiation.

    Returns:
        Dict with LLM tool availability, providers, and feature flags.
    """
    return {
        "llm_tools": {
            "supported": True,
            "tools": list(LLM_TOOL_METADATA.keys()),
            "description": "LLM-powered review, documentation, and fidelity tools",
        },
        "multi_provider": {
            "supported": True,
            "providers": ["cursor-agent", "gemini", "codex"],
            "description": "Multi-provider AI tool integration",
        },
        "data_only_fallback": {
            "supported": True,
            "description": "Graceful degradation when LLM unavailable",
        },
        "feature_flags": {
            name: flag.to_dict() for name, flag in LLM_FEATURE_FLAGS.items()
        },
    }


def is_llm_tool(tool_name: str) -> bool:
    """
    Check if a tool name is an LLM-powered tool.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if tool is an LLM-powered tool
    """
    return tool_name in LLM_TOOL_METADATA


def get_llm_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """
    Get metadata for a specific LLM tool.

    Args:
        tool_name: Name of the LLM tool

    Returns:
        ToolMetadata if found, None otherwise
    """
    return LLM_TOOL_METADATA.get(tool_name)


# =============================================================================
# Provider Tools Discovery Metadata
# =============================================================================


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
