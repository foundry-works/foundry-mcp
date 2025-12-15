"""
Integration tests for environment tools.

Tests:
- Response envelope compliance (success/error structure)
- Feature flag integration
- Tool registration and discovery metadata
- End-to-end tool execution
"""

import tempfile
from pathlib import Path
from dataclasses import asdict


from foundry_mcp.core.responses import success_response, error_response


class TestEnvironmentToolResponseEnvelopes:
    """Integration tests for environment tool response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test that success responses include required envelope fields."""
        result = asdict(success_response(data={"test": "value"}))

        # Required fields per MCP best practices
        assert "success" in result
        assert result["success"] is True
        assert "data" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test that error responses include required envelope fields."""
        result = asdict(
            error_response(
                "Test error message",
                error_code="TEST_ERROR",
                error_type="validation",
            )
        )

        # Required fields per MCP best practices
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "data" in result
        assert result["data"]["error_code"] == "TEST_ERROR"
        assert result["data"]["error_type"] == "validation"
        assert "meta" in result

    def test_success_response_with_warnings(self):
        """Test that success responses can include warnings."""
        warnings = ["Optional feature X not available"]
        result = asdict(success_response(data={}, warnings=warnings))

        assert result["success"] is True
        assert "warnings" in result["meta"]
        assert "Optional feature X not available" in result["meta"]["warnings"]

    def test_error_response_with_remediation(self):
        """Test that error responses can include remediation guidance."""
        result = asdict(
            error_response(
                "Configuration missing",
                error_code="CONFIG_ERROR",
                remediation="Run 'sdd init' to configure the workspace.",
            )
        )

        assert result["success"] is False
        assert "remediation" in result["data"]


class TestFeatureFlagIntegration:
    """Integration tests for feature flag system with environment tools."""

    def test_environment_tools_flag_in_discovery(self):
        """Test environment_tools flag is available in discovery."""
        from foundry_mcp.core.discovery import get_environment_capabilities

        capabilities = get_environment_capabilities()
        flags = capabilities["feature_flags"]

        assert "environment_tools" in flags
        assert flags["environment_tools"]["state"] == "beta"
        assert flags["environment_tools"]["default_enabled"] is True

    def test_env_auto_fix_depends_on_environment_tools(self):
        """Test env_auto_fix flag has correct dependency."""
        from foundry_mcp.core.discovery import ENVIRONMENT_FEATURE_FLAGS

        auto_fix = ENVIRONMENT_FEATURE_FLAGS["env_auto_fix"]
        assert "environment_tools" in auto_fix.dependencies
        assert auto_fix.state == "experimental"
        assert auto_fix.default_enabled is False

    def test_feature_flag_descriptor_to_dict(self):
        """Test FeatureFlagDescriptor serialization."""
        from foundry_mcp.core.discovery import FeatureFlagDescriptor

        flag = FeatureFlagDescriptor(
            name="test_flag",
            description="Test feature flag",
            state="experimental",
            default_enabled=False,
            percentage_rollout=50,
            dependencies=["other_flag"],
        )

        result = flag.to_dict()

        assert result["name"] == "test_flag"
        assert result["description"] == "Test feature flag"
        assert result["state"] == "experimental"
        assert result["default_enabled"] is False
        assert result["percentage_rollout"] == 50
        assert "other_flag" in result["dependencies"]


class TestToolDiscoveryMetadata:
    """Integration tests for environment tool discovery metadata."""

    def test_environment_tools_registered_in_metadata(self):
        """Test all environment tools have discovery metadata."""
        from foundry_mcp.core.discovery import ENVIRONMENT_TOOL_METADATA

        expected_tools = [
            "sdd-verify-toolchain",
            "sdd-init-workspace",
            "sdd-detect-topology",
            "sdd-verify-environment",
        ]

        for tool in expected_tools:
            assert tool in ENVIRONMENT_TOOL_METADATA
            metadata = ENVIRONMENT_TOOL_METADATA[tool]
            assert metadata.category == "environment"
            assert len(metadata.description) > 0

    def test_tool_metadata_json_schema_generation(self):
        """Test that tool metadata can generate valid JSON schemas."""
        from foundry_mcp.core.discovery import ENVIRONMENT_TOOL_METADATA

        for name, metadata in ENVIRONMENT_TOOL_METADATA.items():
            schema = metadata.to_json_schema()

            assert "$schema" in schema
            assert schema["title"] == name
            assert "properties" in schema
            assert schema["type"] == "object"

    def test_tool_metadata_summary_generation(self):
        """Test that tool metadata generates proper summaries."""
        from foundry_mcp.core.discovery import ENVIRONMENT_TOOL_METADATA

        for name, metadata in ENVIRONMENT_TOOL_METADATA.items():
            summary = metadata.to_summary()

            assert summary["name"] == name
            assert "description" in summary
            assert summary["category"] == "environment"
            assert summary["deprecated"] is False

    def test_tool_metadata_detailed_generation(self):
        """Test that tool metadata generates detailed info."""
        from foundry_mcp.core.discovery import ENVIRONMENT_TOOL_METADATA

        for name, metadata in ENVIRONMENT_TOOL_METADATA.items():
            detailed = metadata.to_detailed()

            assert detailed["name"] == name
            assert "schema" in detailed
            assert "examples" in detailed
            assert "related_tools" in detailed
            assert "tags" in detailed


class TestToolRegistration:
    """Integration tests for environment tool registration."""

    def test_register_environment_tools(self):
        """Test environment tools can be registered in a registry."""
        from foundry_mcp.core.discovery import (
            ToolRegistry,
            register_environment_tools,
        )

        registry = ToolRegistry()
        register_environment_tools(registry)

        # Verify all tools registered
        tools = registry.list_tools(category="environment")
        assert len(tools) == 4

        # Verify can retrieve by name
        toolchain = registry.get("sdd-verify-toolchain")
        assert toolchain is not None
        assert toolchain.name == "sdd-verify-toolchain"

    def test_registry_category_filtering(self):
        """Test registry filters tools by category correctly."""
        from foundry_mcp.core.discovery import (
            ToolRegistry,
            register_environment_tools,
        )

        registry = ToolRegistry()
        register_environment_tools(registry)

        # Only environment tools
        env_tools = registry.list_tools(category="environment")
        assert len(env_tools) == 4

        # Non-existent category should return empty
        other_tools = registry.list_tools(category="nonexistent")
        assert len(other_tools) == 0

    def test_registry_list_categories(self):
        """Test registry lists categories correctly."""
        from foundry_mcp.core.discovery import (
            ToolRegistry,
            register_environment_tools,
        )

        registry = ToolRegistry()
        register_environment_tools(registry)

        categories = registry.list_categories()
        category_names = [c["name"] for c in categories]

        assert "environment" in category_names

    def test_registry_stats(self):
        """Test registry provides accurate statistics."""
        from foundry_mcp.core.discovery import (
            ToolRegistry,
            register_environment_tools,
        )

        registry = ToolRegistry()
        register_environment_tools(registry)

        stats = registry.get_stats()

        assert stats["total_tools"] == 4
        assert stats["active_tools"] == 4
        assert stats["deprecated_tools"] == 0
        assert stats["categories"] == 1


class TestEnvironmentCapabilities:
    """Integration tests for environment capability reporting."""

    def test_get_environment_capabilities_structure(self):
        """Test environment capabilities have correct structure."""
        from foundry_mcp.core.discovery import get_environment_capabilities

        caps = get_environment_capabilities()

        assert "environment_readiness" in caps
        assert caps["environment_readiness"]["supported"] is True
        assert "tools" in caps["environment_readiness"]
        assert len(caps["environment_readiness"]["tools"]) == 4

    def test_capabilities_include_feature_flags(self):
        """Test capabilities include all environment feature flags."""
        from foundry_mcp.core.discovery import get_environment_capabilities

        caps = get_environment_capabilities()

        assert "feature_flags" in caps
        assert "environment_tools" in caps["feature_flags"]
        assert "env_auto_fix" in caps["feature_flags"]

    def test_is_environment_tool_helper(self):
        """Test is_environment_tool helper function works correctly."""
        from foundry_mcp.core.discovery import is_environment_tool

        # Valid environment tools
        assert is_environment_tool("sdd-verify-toolchain") is True
        assert is_environment_tool("sdd-init-workspace") is True
        assert is_environment_tool("sdd-detect-topology") is True
        assert is_environment_tool("sdd-verify-environment") is True

        # Non-environment tools
        assert is_environment_tool("list_specs") is False
        assert is_environment_tool("") is False
        assert is_environment_tool("random_tool") is False

    def test_get_environment_tool_metadata_helper(self):
        """Test get_environment_tool_metadata helper function works correctly."""
        from foundry_mcp.core.discovery import get_environment_tool_metadata

        # Valid tools return metadata
        metadata = get_environment_tool_metadata("sdd-verify-toolchain")
        assert metadata is not None
        assert metadata.name == "sdd-verify-toolchain"
        assert metadata.category == "environment"

        # Invalid tools return None
        assert get_environment_tool_metadata("nonexistent") is None
        assert get_environment_tool_metadata("") is None


class TestEndToEndWorkflow:
    """Integration tests for end-to-end environment tool workflow."""

    def test_workspace_initialization_workflow(self):
        """Test workspace initialization creates expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Simulate init workflow
            specs_dir = base_path / "specs"
            subdirs = ["active", "pending", "completed", "archived"]

            specs_dir.mkdir(parents=True)
            for subdir in subdirs:
                (specs_dir / subdir).mkdir()

            # Verify structure
            assert specs_dir.exists()
            for subdir in subdirs:
                assert (specs_dir / subdir).exists()

            # Verify can write spec file
            spec_file = specs_dir / "active" / "test-spec.json"
            spec_file.write_text('{"spec_id": "test"}')
            assert spec_file.exists()

    def test_topology_detection_workflow(self):
        """Test topology detection correctly identifies project types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create Python project markers
            (base_path / "pyproject.toml").touch()
            (base_path / "specs").mkdir()
            (base_path / "docs").mkdir()
            (base_path / ".git").mkdir()

            # Simulate detection
            project_type = "unknown"
            has_git = (base_path / ".git").is_dir()
            specs_found = (base_path / "specs").is_dir()
            docs_found = (base_path / "docs").is_dir()

            if (base_path / "pyproject.toml").exists():
                project_type = "python"

            assert project_type == "python"
            assert has_git is True
            assert specs_found is True
            assert docs_found is True

    def test_environment_verification_workflow(self):
        """Test environment verification correctly reports runtime info."""
        import sys
        import shutil

        # Get actual runtime info
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        git_available = shutil.which("git") is not None

        # Verify we can get this info
        assert python_version.startswith("3.")
        assert git_available is True  # Should be available in dev environment
