"""
Unit tests for foundry_mcp.core.providers.registry module.

Tests cover:
- ProviderRegistration dataclass creation
- register_provider and register_lazy_provider functions
- available_providers listing
- check_provider_available function
- resolve_provider instantiation
- get_provider_metadata function
- describe_providers function
- reset_registry function
"""

from typing import Dict, Optional
from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.providers.base import (
    ProviderCapability,
    ProviderContext,
    ProviderHooks,
    ProviderMetadata,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
)
from foundry_mcp.core.providers.registry import (
    ProviderRegistration,
    register_provider,
    register_lazy_provider,
    available_providers,
    check_provider_available,
    resolve_provider,
    get_provider_metadata,
    describe_providers,
    reset_registry,
    get_registration,
    set_dependency_resolver,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry_fixture():
    """Reset registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


class MockProvider(ProviderContext):
    """Mock provider implementation for testing."""

    def __init__(self, metadata: ProviderMetadata, hooks: ProviderHooks = None):
        super().__init__(metadata, hooks)

    def _execute(self, request: ProviderRequest) -> ProviderResult:
        return ProviderResult(
            content="Mock response",
            provider_id=self._metadata.provider_id,
            model_used=f"{self._metadata.provider_id}:mock",
            status=ProviderStatus.SUCCESS,
        )


def mock_factory(
    *,
    hooks: ProviderHooks,
    model: Optional[str] = None,
    dependencies: Optional[Dict[str, object]] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> MockProvider:
    """Factory function matching ProviderFactory protocol."""
    metadata = ProviderMetadata(provider_id="mock-provider")
    return MockProvider(metadata, hooks)


@pytest.fixture
def test_metadata():
    """Create test provider metadata."""
    return ProviderMetadata(
        provider_id="test-provider",
        display_name="Test Provider",
        capabilities={ProviderCapability.TEXT},
    )


# =============================================================================
# ProviderRegistration Tests
# =============================================================================


class TestProviderRegistration:
    """Tests for ProviderRegistration dataclass."""

    def test_minimal_creation(self):
        """ProviderRegistration should work with required field only."""
        reg = ProviderRegistration(provider_id="test")
        assert reg.provider_id == "test"
        assert reg.factory is None
        assert reg.lazy_loader is None
        assert reg.metadata is None
        assert reg.metadata_resolver is None
        assert reg.availability_check is None
        assert reg.priority == 0
        assert reg.description is None
        assert reg.tags == ()

    def test_full_creation(self, test_metadata):
        """ProviderRegistration should accept all parameters."""
        check = MagicMock(return_value=True)
        reg = ProviderRegistration(
            provider_id="test",
            factory=mock_factory,
            metadata=test_metadata,
            availability_check=check,
            priority=50,
            description="Test Description",
            tags=("tag1", "tag2"),
        )
        assert reg.priority == 50
        assert reg.tags == ("tag1", "tag2")
        assert reg.availability_check == check
        assert reg.description == "Test Description"

    def test_load_factory_with_eager(self, test_metadata):
        """load_factory should return eager factory."""
        reg = ProviderRegistration(
            provider_id="test",
            factory=mock_factory,
        )
        factory = reg.load_factory()
        assert factory == mock_factory

    def test_load_factory_without_factory_raises(self):
        """load_factory without factory should raise."""
        reg = ProviderRegistration(provider_id="test")
        from foundry_mcp.core.providers.base import ProviderUnavailableError
        with pytest.raises(ProviderUnavailableError):
            reg.load_factory()


# =============================================================================
# register_provider Tests
# =============================================================================


class TestRegisterProvider:
    """Tests for register_provider function."""

    def test_register_new_provider(self):
        """Should register a new provider."""
        register_provider(
            provider_id="new-provider",
            factory=mock_factory,
            description="New provider",
        )
        assert "new-provider" in available_providers(include_unavailable=True)

    def test_register_with_priority(self):
        """Should register provider with custom priority."""
        register_provider(
            provider_id="priority-test",
            factory=mock_factory,
            priority=25,
        )
        reg = get_registration("priority-test")
        assert reg is not None
        assert reg.priority == 25

    def test_register_with_tags(self):
        """Should register provider with tags."""
        register_provider(
            provider_id="tagged-provider",
            factory=mock_factory,
            tags=["review", "llm"],
        )
        reg = get_registration("tagged-provider")
        assert reg is not None
        assert "review" in reg.tags
        assert "llm" in reg.tags

    def test_register_duplicate_raises(self):
        """Should raise ValueError for duplicate registration."""
        register_provider(
            provider_id="duplicate-test",
            factory=mock_factory,
        )
        with pytest.raises(ValueError, match="already registered"):
            register_provider(
                provider_id="duplicate-test",
                factory=mock_factory,
            )

    def test_register_with_replace(self):
        """Should allow replacement with replace=True."""
        register_provider(
            provider_id="replace-test",
            factory=mock_factory,
            priority=100,
        )
        register_provider(
            provider_id="replace-test",
            factory=mock_factory,
            priority=50,
            replace=True,
        )
        reg = get_registration("replace-test")
        assert reg.priority == 50


# =============================================================================
# register_lazy_provider Tests
# =============================================================================


class TestRegisterLazyProvider:
    """Tests for register_lazy_provider function."""

    def test_lazy_register_new_provider(self):
        """Should lazily register a new provider."""
        register_lazy_provider(
            provider_id="lazy-test",
            module_path="tests.unit.test_providers_registry",
            factory_attr="mock_factory",
        )
        assert "lazy-test" in available_providers(include_unavailable=True)


# =============================================================================
# available_providers Tests
# =============================================================================


class TestAvailableProviders:
    """Tests for available_providers function."""

    def test_returns_list(self):
        """Should return a list of provider IDs."""
        register_provider(
            provider_id="list-test",
            factory=mock_factory,
        )
        providers = available_providers(include_unavailable=True)
        assert isinstance(providers, list)
        assert "list-test" in providers

    def test_include_unavailable_true(self):
        """include_unavailable=True should return all providers."""
        register_provider(
            provider_id="unavailable-test",
            factory=mock_factory,
        )
        providers = available_providers(include_unavailable=True)
        assert "unavailable-test" in providers


# =============================================================================
# check_provider_available Tests
# =============================================================================


class TestCheckProviderAvailable:
    """Tests for check_provider_available function."""

    def test_unregistered_provider_returns_false(self):
        """Should return False for unregistered provider."""
        result = check_provider_available("nonexistent-provider-xyz")
        assert result is False

    def test_registered_without_check_returns_true(self):
        """Registered provider without availability_check should return True."""
        register_provider(
            provider_id="no-check-test",
            factory=mock_factory,
        )
        result = check_provider_available("no-check-test")
        assert result is True

    def test_with_availability_check_uses_check(self):
        """Should use availability_check if provided."""
        check = MagicMock(return_value=False)
        register_provider(
            provider_id="check-test",
            factory=mock_factory,
            availability_check=check,
        )
        result = check_provider_available("check-test")
        check.assert_called_once()
        assert result is False


# =============================================================================
# resolve_provider Tests
# =============================================================================


class TestResolveProvider:
    """Tests for resolve_provider function."""

    def test_resolve_registered_provider(self):
        """Should resolve a registered provider."""
        register_provider(
            provider_id="resolve-test",
            factory=mock_factory,
        )
        hooks = ProviderHooks()
        provider = resolve_provider("resolve-test", hooks=hooks)
        assert provider is not None
        assert isinstance(provider, MockProvider)

    def test_resolve_unregistered_raises(self):
        """Should raise ProviderUnavailableError for unregistered provider."""
        from foundry_mcp.core.providers.base import ProviderUnavailableError
        hooks = ProviderHooks()
        with pytest.raises(ProviderUnavailableError, match="not registered"):
            resolve_provider("nonexistent-provider-xyz", hooks=hooks)

    def test_resolve_with_hooks(self):
        """Should pass hooks to provider."""
        register_provider(
            provider_id="hooks-test",
            factory=mock_factory,
        )
        hooks = ProviderHooks()
        provider = resolve_provider("hooks-test", hooks=hooks)
        assert provider._hooks == hooks


# =============================================================================
# get_provider_metadata Tests
# =============================================================================


class TestGetProviderMetadata:
    """Tests for get_provider_metadata function."""

    def test_returns_metadata(self, test_metadata):
        """Should return provider metadata."""
        register_provider(
            provider_id="metadata-test",
            factory=mock_factory,
            metadata=test_metadata,
        )
        metadata = get_provider_metadata("metadata-test")
        assert metadata is not None
        assert metadata.provider_id == "test-provider"

    def test_unregistered_returns_none(self):
        """Should return None for unregistered provider."""
        metadata = get_provider_metadata("nonexistent-xyz")
        assert metadata is None


# =============================================================================
# describe_providers Tests
# =============================================================================


class TestDescribeProviders:
    """Tests for describe_providers function."""

    def test_returns_list_of_dicts(self):
        """Should return list of provider descriptions."""
        register_provider(
            provider_id="describe-test",
            factory=mock_factory,
        )
        descriptions = describe_providers()
        assert isinstance(descriptions, list)
        assert len(descriptions) >= 1

    def test_description_contains_expected_fields(self):
        """Each description should contain expected fields."""
        register_provider(
            provider_id="fields-test",
            factory=mock_factory,
            priority=75,
            tags=["test-tag"],
            description="Test description",
        )
        descriptions = describe_providers()
        desc = next((d for d in descriptions if d["id"] == "fields-test"), None)
        assert desc is not None
        assert "id" in desc
        assert "description" in desc
        assert "priority" in desc
        assert "tags" in desc
        assert "available" in desc


# =============================================================================
# reset_registry Tests
# =============================================================================


class TestResetRegistry:
    """Tests for reset_registry function."""

    def test_clears_registered_providers(self):
        """reset should clear all registered providers."""
        register_provider(
            provider_id="reset-test",
            factory=mock_factory,
        )
        assert "reset-test" in available_providers(include_unavailable=True)
        reset_registry()
        assert "reset-test" not in available_providers(include_unavailable=True)


# =============================================================================
# get_registration Tests
# =============================================================================


class TestGetRegistration:
    """Tests for get_registration function."""

    def test_returns_registration(self):
        """Should return registration for registered provider."""
        register_provider(
            provider_id="get-reg-test",
            factory=mock_factory,
        )
        reg = get_registration("get-reg-test")
        assert reg is not None
        assert reg.provider_id == "get-reg-test"

    def test_unregistered_returns_none(self):
        """Should return None for unregistered provider."""
        reg = get_registration("nonexistent-xyz")
        assert reg is None


# =============================================================================
# Dependency Resolver Tests
# =============================================================================


class TestDependencyResolver:
    """Tests for dependency resolution."""

    def test_set_dependency_resolver(self):
        """Should set custom dependency resolver."""
        resolver = MagicMock(return_value={"custom": "deps"})
        set_dependency_resolver(resolver)
        # Reset to avoid affecting other tests
        set_dependency_resolver(None)

    def test_clear_dependency_resolver(self):
        """Should clear dependency resolver with None."""
        resolver = MagicMock()
        set_dependency_resolver(resolver)
        set_dependency_resolver(None)
        # Should not raise
