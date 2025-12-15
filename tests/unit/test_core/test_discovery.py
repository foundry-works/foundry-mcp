"""Tests for discovery module."""

import pytest

from foundry_mcp.core.discovery import (
    SCHEMA_VERSION,
    ParameterType,
    ParameterMetadata,
    ToolMetadata,
    ToolRegistry,
    ServerCapabilities,
    get_capabilities,
    negotiate_capabilities,
    deprecated_tool,
    is_deprecated,
    get_deprecation_info,
)


class TestParameterMetadata:
    """Tests for ParameterMetadata dataclass."""

    def test_create_basic_parameter(self):
        """Should create basic parameter with defaults."""
        param = ParameterMetadata(
            name="user_id",
            type=ParameterType.STRING,
            description="User identifier",
        )
        assert param.name == "user_id"
        assert param.type == ParameterType.STRING
        assert param.description == "User identifier"
        assert param.required is False
        assert param.default is None
        assert param.constraints == {}
        assert param.examples == []

    def test_create_required_parameter(self):
        """Should create required parameter."""
        param = ParameterMetadata(
            name="email",
            type=ParameterType.STRING,
            description="Email address",
            required=True,
            constraints={"format": "email"},
        )
        assert param.required is True
        assert param.constraints == {"format": "email"}

    def test_create_parameter_with_examples(self):
        """Should create parameter with examples."""
        param = ParameterMetadata(
            name="count",
            type=ParameterType.INTEGER,
            description="Number of items",
            default=10,
            examples=[5, 10, 20],
        )
        assert param.default == 10
        assert param.examples == [5, 10, 20]


class TestToolMetadata:
    """Tests for ToolMetadata dataclass."""

    def test_create_basic_tool(self):
        """Should create basic tool with defaults."""
        tool = ToolMetadata(
            name="get_user",
            description="Get user by ID",
        )
        assert tool.name == "get_user"
        assert tool.description == "Get user by ID"
        assert tool.parameters == []
        assert tool.category == "general"
        assert tool.version == "1.0.0"
        assert tool.deprecated is False
        assert tool.tags == []

    def test_create_tool_with_parameters(self):
        """Should create tool with parameters."""
        tool = ToolMetadata(
            name="create_user",
            description="Create new user",
            parameters=[
                ParameterMetadata(
                    name="email",
                    type=ParameterType.STRING,
                    description="Email",
                    required=True,
                ),
                ParameterMetadata(
                    name="name",
                    type=ParameterType.STRING,
                    description="Name",
                ),
            ],
            category="users",
            tags=["write", "users"],
        )
        assert len(tool.parameters) == 2
        assert tool.category == "users"
        assert tool.tags == ["write", "users"]

    def test_to_json_schema_basic(self):
        """Should generate basic JSON schema."""
        tool = ToolMetadata(
            name="test_tool",
            description="Test tool",
        )
        schema = tool.to_json_schema()

        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "test_tool"
        assert schema["description"] == "Test tool"
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []
        assert schema["additionalProperties"] is False

    def test_to_json_schema_with_parameters(self):
        """Should generate JSON schema with parameters."""
        tool = ToolMetadata(
            name="search",
            description="Search items",
            parameters=[
                ParameterMetadata(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ParameterMetadata(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Max results",
                    default=10,
                    constraints={"minimum": 1, "maximum": 100},
                ),
            ],
        )
        schema = tool.to_json_schema()

        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["query"]["description"] == "Search query"

        assert "limit" in schema["properties"]
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["limit"]["default"] == 10
        assert schema["properties"]["limit"]["minimum"] == 1
        assert schema["properties"]["limit"]["maximum"] == 100

        assert schema["required"] == ["query"]

    def test_to_summary(self):
        """Should generate summary with truncated description."""
        tool = ToolMetadata(
            name="test",
            description="A" * 250,  # Long description
            category="test",
            version="1.0.0",
            tags=["test"],
        )
        summary = tool.to_summary()

        assert summary["name"] == "test"
        assert len(summary["description"]) == 200
        assert summary["description"].endswith("...")
        assert summary["category"] == "test"
        assert summary["version"] == "1.0.0"
        assert summary["deprecated"] is False
        assert summary["tags"] == ["test"]

    def test_to_detailed(self):
        """Should generate detailed tool info."""
        tool = ToolMetadata(
            name="detailed_tool",
            description="Detailed tool",
            category="test",
            version="2.0.0",
            rate_limit="100/minute",
            examples=[{"input": {"a": 1}, "output": {"b": 2}}],
            related_tools=["other_tool"],
            tags=["read"],
            deprecated=True,
            deprecation_message="Use other_tool instead",
        )
        detailed = tool.to_detailed()

        assert detailed["name"] == "detailed_tool"
        assert detailed["version"] == "2.0.0"
        assert detailed["category"] == "test"
        assert "schema" in detailed
        assert detailed["examples"] == [{"input": {"a": 1}, "output": {"b": 2}}]
        assert detailed["related_tools"] == ["other_tool"]
        assert detailed["rate_limit"] == "100/minute"
        assert detailed["deprecated"] is True
        assert detailed["deprecation_message"] == "Use other_tool instead"


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self):
        """Should register tool in registry."""
        registry = ToolRegistry()
        tool = ToolMetadata(name="test_tool", description="Test")
        registry.register(tool)

        assert registry.get("test_tool") is not None
        assert registry.get("test_tool").name == "test_tool"

    def test_register_duplicate_raises(self):
        """Should raise error for duplicate registration."""
        registry = ToolRegistry()
        tool = ToolMetadata(name="duplicate", description="Test")
        registry.register(tool)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_unregister_tool(self):
        """Should unregister tool from registry."""
        registry = ToolRegistry()
        tool = ToolMetadata(name="removable", description="Test")
        registry.register(tool)

        assert registry.unregister("removable") is True
        assert registry.get("removable") is None

    def test_unregister_nonexistent(self):
        """Should return False for nonexistent tool."""
        registry = ToolRegistry()
        assert registry.unregister("nonexistent") is False

    def test_list_tools_all(self):
        """Should list all tools."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="tool1", description="Test 1"))
        registry.register(ToolMetadata(name="tool2", description="Test 2"))

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_list_tools_by_category(self):
        """Should filter tools by category."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="user_get", description="Get user", category="users"))
        registry.register(ToolMetadata(name="file_read", description="Read file", category="files"))

        tools = registry.list_tools(category="users")
        assert len(tools) == 1
        assert tools[0]["name"] == "user_get"

    def test_list_tools_by_tag(self):
        """Should filter tools by tag."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="reader", description="Read", tags=["read"]))
        registry.register(ToolMetadata(name="writer", description="Write", tags=["write"]))

        tools = registry.list_tools(tag="read")
        assert len(tools) == 1
        assert tools[0]["name"] == "reader"

    def test_list_tools_excludes_deprecated(self):
        """Should exclude deprecated tools by default."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="active", description="Active"))
        registry.register(ToolMetadata(name="old", description="Old", deprecated=True))

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "active"

    def test_list_tools_include_deprecated(self):
        """Should include deprecated tools when requested."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="active", description="Active"))
        registry.register(ToolMetadata(name="old", description="Old", deprecated=True))

        tools = registry.list_tools(include_deprecated=True)
        assert len(tools) == 2

    def test_get_tool_schema(self):
        """Should return detailed schema for tool."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="schema_test", description="Test"))

        schema = registry.get_tool_schema("schema_test")
        assert schema is not None
        assert schema["name"] == "schema_test"
        assert "schema" in schema

    def test_get_tool_schema_not_found(self):
        """Should return None for unknown tool."""
        registry = ToolRegistry()
        assert registry.get_tool_schema("unknown") is None

    def test_list_categories(self):
        """Should list categories with counts."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="user1", description="U1", category="users"))
        registry.register(ToolMetadata(name="user2", description="U2", category="users"))
        registry.register(ToolMetadata(name="file1", description="F1", category="files"))

        categories = registry.list_categories()
        assert len(categories) == 2

        users_cat = next(c for c in categories if c["name"] == "users")
        assert users_cat["tool_count"] == 2

    def test_get_stats(self):
        """Should return registry statistics."""
        registry = ToolRegistry()
        registry.register(ToolMetadata(name="active", description="A"))
        registry.register(ToolMetadata(name="deprecated", description="D", deprecated=True))

        stats = registry.get_stats()
        assert stats["total_tools"] == 2
        assert stats["active_tools"] == 1
        assert stats["deprecated_tools"] == 1
        assert stats["categories"] == 1  # Both default to "general"


class TestServerCapabilities:
    """Tests for ServerCapabilities dataclass."""

    def test_default_capabilities(self):
        """Should have sensible defaults."""
        caps = ServerCapabilities()
        assert caps.response_version == "response-v2"
        assert caps.supports_streaming is False
        assert caps.supports_batch is True
        assert caps.supports_pagination is True
        assert caps.max_batch_size == 100
        assert caps.rate_limit_headers is True
        assert caps.supported_formats == ["json"]
        assert caps.feature_flags_enabled is True

    def test_to_dict(self):
        """Should convert to dict."""
        caps = ServerCapabilities()
        d = caps.to_dict()

        assert d["response_version"] == "response-v2"
        assert d["streaming"] is False
        assert d["batch_operations"] is True
        assert d["pagination"] is True
        assert d["max_batch_size"] == 100
        assert d["rate_limit_headers"] is True
        assert d["formats"] == ["json"]
        assert d["feature_flags"] is True


class TestGetCapabilities:
    """Tests for get_capabilities function."""

    def test_returns_capabilities(self):
        """Should return capabilities dict."""
        caps = get_capabilities()

        assert "schema_version" in caps
        assert caps["schema_version"] == SCHEMA_VERSION
        assert "capabilities" in caps
        assert "server_version" in caps
        assert "api_version" in caps


class TestNegotiateCapabilities:
    """Tests for negotiate_capabilities function."""

    def test_negotiate_matching_version(self):
        """Should accept matching version."""
        result = negotiate_capabilities(requested_version="response-v2")

        assert result["negotiated"]["response_version"] == "response-v2"
        assert result["warnings"] is None

    def test_negotiate_mismatched_version(self):
        """Should warn about mismatched version."""
        result = negotiate_capabilities(requested_version="response-v1")

        assert result["negotiated"]["response_version"] == "response-v2"
        assert result["warnings"] is not None
        assert any("not supported" in w for w in result["warnings"])

    def test_negotiate_features(self):
        """Should negotiate requested features."""
        result = negotiate_capabilities(requested_features=["batch", "pagination"])

        assert result["negotiated"]["features"]["batch"] is True
        assert result["negotiated"]["features"]["pagination"] is True

    def test_negotiate_unknown_feature(self):
        """Should warn about unknown features."""
        result = negotiate_capabilities(requested_features=["unknown_feature"])

        assert result["negotiated"]["features"]["unknown_feature"] is False
        assert result["warnings"] is not None
        assert any("not recognized" in w for w in result["warnings"])


class TestDeprecatedToolDecorator:
    """Tests for @deprecated_tool decorator."""

    def test_decorator_updates_docstring(self):
        """Should update function docstring."""
        @deprecated_tool(replacement="new_func", removal_version="2.0.0")
        def old_func():
            """Original docstring."""
            return {"data": "test"}

        assert "[DEPRECATED]" in old_func.__doc__
        assert "2.0.0" in old_func.__doc__
        assert "new_func" in old_func.__doc__

    def test_decorator_adds_warning_to_response(self):
        """Should add deprecation warning to response meta."""
        @deprecated_tool(replacement="new_func", removal_version="2.0.0")
        def old_func():
            return {"data": "test", "meta": {"version": "response-v2"}}

        result = old_func()
        assert "warnings" in result["meta"]
        assert any("DEPRECATED" in w for w in result["meta"]["warnings"])

    def test_decorator_creates_meta_if_missing(self):
        """Should create meta if not present."""
        @deprecated_tool(replacement="new_func", removal_version="2.0.0")
        def old_func():
            return {"data": "test"}

        result = old_func()
        assert "meta" in result
        assert "warnings" in result["meta"]

    def test_is_deprecated_helper(self):
        """Should detect deprecated functions."""
        @deprecated_tool(replacement="new", removal_version="2.0.0")
        def deprecated_func():
            return {}

        def normal_func():
            return {}

        assert is_deprecated(deprecated_func) is True
        assert is_deprecated(normal_func) is False

    def test_get_deprecation_info(self):
        """Should return deprecation info."""
        @deprecated_tool(replacement="replacement_func", removal_version="3.0.0")
        def old_func():
            return {}

        info = get_deprecation_info(old_func)
        assert info is not None
        assert info["replacement"] == "replacement_func"
        assert info["removal_version"] == "3.0.0"

    def test_get_deprecation_info_not_deprecated(self):
        """Should return None for non-deprecated functions."""
        def normal_func():
            return {}

        assert get_deprecation_info(normal_func) is None
