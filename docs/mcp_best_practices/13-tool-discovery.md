# 13. Tool Metadata & Discovery

> Enable clients to discover tools, understand capabilities, and negotiate features.

## Overview

MCP's power comes from dynamic tool discovery. Clients need to understand what tools exist, what they do, and how to use them correctly. Well-designed metadata enables effective tool selection and invocation.

## Requirements

### MUST

- **Provide machine-readable schemas** - for all tool inputs/outputs
- **Write clear descriptions** - that help LLMs select appropriate tools
- **Document all parameters** - required, optional, defaults, constraints
- **Version tool interfaces** - track breaking changes

### SHOULD

- **Include usage examples** - in tool descriptions
- **Group related tools** - by category or domain
- **Document rate limits** - per-tool constraints
- **Provide input validation schemas** - JSON Schema format
- **Follow canonical naming** - apply the prefixes in [Tool Naming Conventions](../codebase_standards/naming-conventions.md) so registries stay predictable

### MAY

- **Support capability negotiation** - client feature detection
- **Include semantic tags** - for tool categorization
- **Provide deprecation notices** - for tools being phased out

## Tool Registration

### Complete Tool Definition

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"

@dataclass
class Parameter:
    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: List[Parameter]
    category: str = "general"
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: str = None
    rate_limit: str = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

# Example tool definition
get_user_tool = ToolDefinition(
    name="get_user",
    description="""Retrieve user details by ID or email.

    Returns the user's profile information including name, email,
    and account status. Use this to look up specific users before
    performing operations on their data.

    For listing multiple users, use list_users instead.""",

    parameters=[
        Parameter(
            name="user_id",
            type=ParameterType.STRING,
            description="Unique user identifier (starts with 'usr_')",
            required=False,
            constraints={"pattern": "^usr_[a-zA-Z0-9]+$"},
            examples=["usr_abc123", "usr_xyz789"]
        ),
        Parameter(
            name="email",
            type=ParameterType.STRING,
            description="User's email address (alternative to user_id)",
            required=False,
            constraints={"format": "email"},
            examples=["alice@example.com"]
        ),
        Parameter(
            name="include_profile",
            type=ParameterType.BOOLEAN,
            description="Include extended profile data",
            required=False,
            default=False
        )
    ],

    category="users",
    version="2.1.0",
    rate_limit="100/minute",

    examples=[
        {
            "description": "Get user by ID",
            "input": {"user_id": "usr_abc123"},
            "output": {"success": True, "data": {"user": {"id": "usr_abc123", "name": "Alice"}}}
        },
        {
            "description": "Get user by email with profile",
            "input": {"email": "alice@example.com", "include_profile": True},
            "output": {"success": True, "data": {"user": {"id": "usr_abc123", "name": "Alice", "profile": {...}}}}
        }
    ],

    related_tools=["list_users", "update_user", "delete_user"],
    tags=["users", "read", "lookup"]
)
```

## JSON Schema Generation

```python
def tool_to_json_schema(tool: ToolDefinition) -> dict:
    """Generate JSON Schema from tool definition."""

    type_mapping = {
        ParameterType.STRING: "string",
        ParameterType.INTEGER: "integer",
        ParameterType.BOOLEAN: "boolean",
        ParameterType.ARRAY: "array",
        ParameterType.OBJECT: "object",
    }

    properties = {}
    required = []

    for param in tool.parameters:
        prop = {
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
        "title": tool.name,
        "description": tool.description,
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }
```

## Tool Discovery Endpoint

```python
@mcp.tool()
def list_tools(
    category: str = None,
    tag: str = None,
    include_deprecated: bool = False
) -> dict:
    """List available tools with filtering.

    Args:
        category: Filter by category (e.g., "users", "files")
        tag: Filter by tag (e.g., "read", "write")
        include_deprecated: Include deprecated tools
    """
    tools = get_registered_tools()

    # Apply filters
    if category:
        tools = [t for t in tools if t.category == category]
    if tag:
        tools = [t for t in tools if tag in t.tags]
    if not include_deprecated:
        tools = [t for t in tools if not t.deprecated]

    return asdict(success_response(data={
        "tools": [
            {
                "name": t.name,
                "description": t.description[:200] + "..." if len(t.description) > 200 else t.description,
                "category": t.category,
                "version": t.version,
                "deprecated": t.deprecated,
                "tags": t.tags
            }
            for t in tools
        ],
        "categories": list(set(t.category for t in tools)),
        "total": len(tools)
    }))

@mcp.tool()
def get_tool_schema(tool_name: str) -> dict:
    """Get detailed schema for a specific tool.

    Returns full JSON Schema, examples, and related tools.
    """
    tool = get_tool_by_name(tool_name)
    if not tool:
        return asdict(error_response(f"Tool '{tool_name}' not found"))

    schema = tool_to_json_schema(tool)

    return asdict(success_response(data={
        "name": tool.name,
        "version": tool.version,
        "description": tool.description,
        "schema": schema,
        "examples": tool.examples,
        "related_tools": tool.related_tools,
        "rate_limit": tool.rate_limit,
        "deprecated": tool.deprecated,
        "deprecation_message": tool.deprecation_message
    }))
```

## Capability Negotiation

```python
from typing import Set

@dataclass
class ServerCapabilities:
    """Capabilities supported by this MCP server."""
    response_version: str = "response-v2"
    supports_streaming: bool = False
    supports_batch: bool = True
    supports_pagination: bool = True
    max_batch_size: int = 100
    rate_limit_headers: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ["json"])

@mcp.tool()
def get_capabilities() -> dict:
    """Get server capabilities for client negotiation.

    Clients should call this to understand server features
    before making assumptions about available functionality.
    """
    caps = ServerCapabilities()

    return asdict(success_response(data={
        "capabilities": {
            "response_version": caps.response_version,
            "streaming": caps.supports_streaming,
            "batch_operations": caps.supports_batch,
            "pagination": caps.supports_pagination,
            "max_batch_size": caps.max_batch_size,
            "rate_limit_headers": caps.rate_limit_headers,
            "formats": caps.supported_formats
        },
        "server_version": "1.5.0",
        "api_version": "2024-11-01"
    }))

@mcp.tool()
def negotiate_capabilities(
    requested_version: str = None,
    requested_features: List[str] = None
) -> dict:
    """Negotiate capabilities with client.

    Args:
        requested_version: Desired response version
        requested_features: List of requested features
    """
    server_caps = ServerCapabilities()
    negotiated = {}
    warnings = []

    # Version negotiation
    if requested_version:
        if requested_version == server_caps.response_version:
            negotiated["response_version"] = requested_version
        else:
            negotiated["response_version"] = server_caps.response_version
            warnings.append(
                f"Requested version '{requested_version}' not supported, "
                f"using '{server_caps.response_version}'"
            )

    # Feature negotiation
    available_features = {
        "streaming": server_caps.supports_streaming,
        "batch": server_caps.supports_batch,
        "pagination": server_caps.supports_pagination,
    }

    if requested_features:
        negotiated["features"] = {}
        for feature in requested_features:
            if feature in available_features:
                negotiated["features"][feature] = available_features[feature]
            else:
                negotiated["features"][feature] = False
                warnings.append(f"Feature '{feature}' not recognized")

    return asdict(success_response(
        data={"negotiated": negotiated},
        warnings=warnings if warnings else None
    ))
```

## Tool Descriptions for LLMs

### Effective Description Format

```python
@mcp.tool()
def search_codebase(
    query: str,
    file_pattern: str = "*",
    max_results: int = 20,
    context_lines: int = 2
) -> dict:
    """Search codebase for text patterns.

    Use this tool when you need to find code containing specific text,
    function names, variable names, or error messages. Returns matching
    lines with surrounding context.

    WHEN TO USE:
    - Finding where a function is defined or called
    - Locating error messages or log statements
    - Searching for specific patterns or strings
    - Discovering usage examples of APIs

    WHEN NOT TO USE:
    - Listing files (use list_files instead)
    - Reading entire files (use read_file instead)
    - Searching by file name (use find_files instead)

    Args:
        query: Search pattern (supports regex)
        file_pattern: Glob pattern to filter files (e.g., "*.py", "src/**/*.ts")
        max_results: Maximum matches to return (default: 20, max: 100)
        context_lines: Lines of context around each match (default: 2)

    Returns:
        List of matches with file path, line number, and context.

    Example:
        search_codebase("def authenticate", "*.py")
        -> Finds all Python files containing "def authenticate"
    """
    ...
```

### Category Organization

```python
TOOL_CATEGORIES = {
    "files": {
        "description": "File system operations",
        "tools": ["read_file", "write_file", "list_files", "find_files", "delete_file"]
    },
    "code": {
        "description": "Code analysis and manipulation",
        "tools": ["search_codebase", "analyze_code", "format_code", "lint_code"]
    },
    "git": {
        "description": "Version control operations",
        "tools": ["git_status", "git_diff", "git_commit", "git_branch", "git_log"]
    },
    "users": {
        "description": "User management",
        "tools": ["get_user", "list_users", "create_user", "update_user", "delete_user"]
    }
}

@mcp.tool()
def list_categories() -> dict:
    """List tool categories with descriptions.

    Helps understand the organization of available tools.
    """
    return asdict(success_response(data={
        "categories": [
            {
                "name": name,
                "description": info["description"],
                "tool_count": len(info["tools"])
            }
            for name, info in TOOL_CATEGORIES.items()
        ]
    }))
```

## Deprecation Handling

```python
def deprecated_tool(replacement: str, removal_version: str):
    """Decorator to mark tools as deprecated."""
    def decorator(func):
        original_doc = func.__doc__ or ""

        func.__doc__ = f"""[DEPRECATED] {original_doc}

        ⚠️  This tool is deprecated and will be removed in version {removal_version}.
        Use '{replacement}' instead.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Add deprecation warning to response
            if isinstance(result, dict) and "meta" in result:
                if "warnings" not in result["meta"]:
                    result["meta"]["warnings"] = []
                result["meta"]["warnings"].append(
                    f"DEPRECATED: '{func.__name__}' will be removed in {removal_version}. "
                    f"Use '{replacement}' instead."
                )

            return result
        return wrapper
    return decorator

# Usage
@mcp.tool()
@deprecated_tool(replacement="get_user", removal_version="3.0.0")
def fetch_user(user_id: str) -> dict:
    """Fetch user by ID (deprecated)."""
    return get_user(user_id)
```

## Rate Limit Documentation

```python
@dataclass
class RateLimitInfo:
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    applies_to: str  # "per_user", "per_tool", "global"

TOOL_RATE_LIMITS = {
    "search_codebase": RateLimitInfo(60, 1000, 10, "per_user"),
    "write_file": RateLimitInfo(30, 500, 5, "per_user"),
    "get_user": RateLimitInfo(100, 5000, 20, "per_user"),
}

@mcp.tool()
def get_rate_limits(tool_name: str = None) -> dict:
    """Get rate limit information for tools.

    Args:
        tool_name: Specific tool to query, or None for all tools
    """
    if tool_name:
        if tool_name not in TOOL_RATE_LIMITS:
            return asdict(error_response(f"Unknown tool: {tool_name}"))

        limit = TOOL_RATE_LIMITS[tool_name]
        return asdict(success_response(data={
            "tool": tool_name,
            "limits": {
                "requests_per_minute": limit.requests_per_minute,
                "requests_per_hour": limit.requests_per_hour,
                "burst_limit": limit.burst_limit,
                "applies_to": limit.applies_to
            }
        }))

    return asdict(success_response(data={
        "rate_limits": {
            name: {
                "requests_per_minute": limit.requests_per_minute,
                "requests_per_hour": limit.requests_per_hour,
                "burst_limit": limit.burst_limit,
                "applies_to": limit.applies_to
            }
            for name, limit in TOOL_RATE_LIMITS.items()
        }
    }))
```

## Anti-Patterns

### Don't: Vague Descriptions

```python
# Bad
@mcp.tool()
def process(data):
    """Process the data."""
    ...

# Good
@mcp.tool()
def validate_email_addresses(emails: List[str]) -> dict:
    """Validate a list of email addresses for correct format.

    Checks each email against RFC 5322 format requirements.
    Returns valid emails and details about invalid ones.
    """
    ...
```

### Don't: Missing Parameter Documentation

```python
# Bad
@mcp.tool()
def search(q, n=10, f=None):
    """Search for stuff."""
    ...

# Good
@mcp.tool()
def search_documents(
    query: str,
    max_results: int = 10,
    filter_type: str = None
) -> dict:
    """Search documents by text content.

    Args:
        query: Search query (supports AND, OR, NOT operators)
        max_results: Maximum documents to return (1-100, default: 10)
        filter_type: Filter by document type ("pdf", "doc", "txt", or None for all)
    """
    ...
```

### Don't: Hide Breaking Changes

```python
# Bad: Silent breaking change
@mcp.tool()
def get_user(user_id: str):  # Was: def get_user(id: str)
    ...

# Good: Version and deprecate
@mcp.tool()  # v2.0.0
def get_user(user_id: str):
    """Get user by ID.

    Note: Parameter renamed from 'id' to 'user_id' in v2.0.0
    """
    ...
```

## Related Documents

- [Versioned Contracts](./01-versioned-contracts.md) - Interface versioning
- [AI/LLM Integration](./11-ai-llm-integration.md) - LLM-friendly descriptions
- [Spec-Driven Development](./09-spec-driven-development.md) - Spec-first design

---

**Navigation:** [← Timeout & Resilience](./12-timeout-resilience.md) | [Index](./README.md) | [Next: Feature Flags →](./14-feature-flags.md)
