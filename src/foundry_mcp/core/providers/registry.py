"""
Provider registry utilities.

Encapsulates registration, lazy loading, availability checks, and dependency
injection hooks for ProviderContext implementations. This module backs the
CLI provider runner plus future skill integrations, providing a single source
of truth for discovering available providers.

Example:
    >>> from foundry_mcp.core.providers.registry import (
    ...     register_provider,
    ...     resolve_provider,
    ...     available_providers,
    ... )
    >>> from foundry_mcp.core.providers import ProviderHooks
    >>>
    >>> # Register a provider factory
    >>> register_provider(
    ...     "my-provider",
    ...     factory=my_provider_factory,
    ...     description="My custom provider",
    ... )
    >>>
    >>> # List available providers
    >>> available_providers()
    ['my-provider']
    >>>
    >>> # Resolve and instantiate a provider
    >>> hooks = ProviderHooks()
    >>> provider = resolve_provider("my-provider", hooks=hooks)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from foundry_mcp.core.providers.base import (
    ProviderContext,
    ProviderHooks,
    ProviderMetadata,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


class ProviderFactory(Protocol):
    """
    Callable that instantiates a ProviderContext.

    Implementations should accept keyword-only arguments for hooks, model,
    dependencies, and overrides so the registry can pass future options
    without breaking signatures.

    Example:
        def create_my_provider(
            *,
            hooks: ProviderHooks,
            model: Optional[str] = None,
            dependencies: Optional[Dict[str, object]] = None,
            overrides: Optional[Dict[str, object]] = None,
        ) -> ProviderContext:
            return MyProvider(hooks=hooks, model=model)
    """

    def __call__(
        self,
        *,
        hooks: ProviderHooks,
        model: Optional[str] = None,
        dependencies: Optional[Dict[str, object]] = None,
        overrides: Optional[Dict[str, object]] = None,
    ) -> ProviderContext:
        ...


# Type aliases for registry callables
AvailabilityCheck = Callable[[], bool]
MetadataResolver = Callable[[], ProviderMetadata]
LazyFactoryLoader = Callable[[], ProviderFactory]
DependencyResolver = Callable[[str], Dict[str, object]]


# =============================================================================
# Provider Registration
# =============================================================================


@dataclass
class ProviderRegistration:
    """
    Internal record for a registered provider.

    Supports both eager and lazy factory loading, with optional metadata
    resolvers and availability checks.

    Attributes:
        provider_id: Canonical provider identifier (e.g., "gemini", "codex")
        factory: Callable that instantiates ProviderContext (eager)
        lazy_loader: Callable that returns a factory when invoked (lazy import)
        metadata: Cached ProviderMetadata object
        metadata_resolver: Callable that returns ProviderMetadata on demand
        availability_check: Callable returning bool to gate resolution
        priority: Sorting priority for available_providers (higher first)
        description: Human-readable description for diagnostics
        tags: Optional labels describing the provider (e.g., ["cli", "external"])
    """

    provider_id: str
    factory: Optional[ProviderFactory] = None
    lazy_loader: Optional[LazyFactoryLoader] = None
    metadata: Optional[ProviderMetadata] = None
    metadata_resolver: Optional[MetadataResolver] = None
    availability_check: Optional[AvailabilityCheck] = None
    priority: int = 0
    description: Optional[str] = None
    tags: Sequence[str] = field(default_factory=tuple)

    def load_factory(self) -> ProviderFactory:
        """
        Return the provider factory, performing lazy import if needed.

        Returns:
            The provider factory callable

        Raises:
            ProviderUnavailableError: If no factory is available
        """
        if self.factory is not None:
            return self.factory

        if self.lazy_loader is None:
            raise ProviderUnavailableError(
                f"Provider '{self.provider_id}' is missing a factory.",
                provider=self.provider_id,
            )

        factory = self.lazy_loader()
        if factory is None:
            raise ProviderUnavailableError(
                f"Lazy loader for '{self.provider_id}' returned None.",
                provider=self.provider_id,
            )

        # Cache the factory for future calls
        self.factory = factory
        self.lazy_loader = None
        return factory

    def is_available(self) -> bool:
        """
        Return True if the provider passes its availability check.

        If no availability_check is registered, returns True.
        """
        if self.availability_check is None:
            return True
        try:
            return bool(self.availability_check())
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Availability check for provider '%s' failed: %s",
                self.provider_id,
                exc,
            )
            return False

    def resolve_metadata(self) -> Optional[ProviderMetadata]:
        """
        Return cached metadata, resolving lazily when necessary.

        Returns:
            ProviderMetadata if available, None otherwise
        """
        if self.metadata is not None:
            return self.metadata

        if self.metadata_resolver is None:
            return None

        try:
            self.metadata = self.metadata_resolver()
            return self.metadata
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Metadata resolver for provider '%s' failed: %s",
                self.provider_id,
                exc,
            )
            return None


# Global registry state
_REGISTRY: Dict[str, ProviderRegistration] = {}
_dependency_resolver: Optional[DependencyResolver] = None


# =============================================================================
# Public API - Registration
# =============================================================================


def register_provider(
    provider_id: str,
    *,
    factory: Optional[ProviderFactory] = None,
    lazy_loader: Optional[LazyFactoryLoader] = None,
    metadata: Optional[ProviderMetadata] = None,
    metadata_resolver: Optional[MetadataResolver] = None,
    availability_check: Optional[AvailabilityCheck] = None,
    priority: int = 0,
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    replace: bool = False,
) -> None:
    """
    Register a provider factory with the global registry.

    Args:
        provider_id: Canonical provider identifier (e.g., "gemini")
        factory: Callable that instantiates ProviderContext instances
        lazy_loader: Callable that returns a factory when invoked (lazy import)
        metadata: Optional ProviderMetadata object (cached)
        metadata_resolver: Callable that returns ProviderMetadata on demand
        availability_check: Callable returning bool to gate resolution
        priority: Sorting priority for available_providers (higher first)
        description: Human-readable description for diagnostics
        tags: Optional labels describing the provider
        replace: Overwrite existing registration if True

    Raises:
        ValueError: If provider_id already registered and replace=False
        ValueError: If neither factory nor lazy_loader provided

    Example:
        >>> register_provider(
        ...     "my-provider",
        ...     factory=my_factory,
        ...     availability_check=lambda: True,
        ...     priority=10,
        ...     description="My custom provider",
        ...     tags=["external", "experimental"],
        ... )
    """
    if provider_id in _REGISTRY and not replace:
        raise ValueError(f"Provider '{provider_id}' is already registered")

    if factory is None and lazy_loader is None:
        raise ValueError("Either 'factory' or 'lazy_loader' must be provided")

    registration = ProviderRegistration(
        provider_id=provider_id,
        factory=factory,
        lazy_loader=lazy_loader,
        metadata=metadata,
        metadata_resolver=metadata_resolver,
        availability_check=availability_check,
        priority=priority,
        description=description,
        tags=tuple(tags or ()),
    )
    _REGISTRY[provider_id] = registration
    logger.debug("Provider '%s' registered (priority=%s)", provider_id, priority)


def register_lazy_provider(
    provider_id: str,
    module_path: str,
    *,
    factory_attr: str = "create_provider",
    metadata_attr: Optional[str] = None,
    availability_attr: Optional[str] = None,
    priority: int = 0,
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    replace: bool = False,
) -> None:
    """
    Register a provider by module path without importing upfront.

    This is useful for providers with heavy dependencies that should only
    be loaded when actually needed.

    Args:
        provider_id: Canonical provider identifier
        module_path: Full module path (e.g., "mypackage.providers.gemini")
        factory_attr: Attribute name for factory function (default: "create_provider")
        metadata_attr: Attribute name for metadata (callable or static)
        availability_attr: Attribute name for availability check (callable)
        priority: Sorting priority for available_providers (higher first)
        description: Human-readable description
        tags: Optional labels
        replace: Overwrite existing registration if True

    Example:
        >>> register_lazy_provider(
        ...     "gemini",
        ...     "foundry_mcp.providers.gemini",
        ...     factory_attr="create_gemini_provider",
        ...     availability_attr="is_gemini_available",
        ... )
    """

    def _lazy_loader() -> ProviderFactory:
        module = importlib.import_module(module_path)
        factory_obj = getattr(module, factory_attr, None)
        if factory_obj is None:
            raise ProviderUnavailableError(
                f"Module '{module_path}' is missing '{factory_attr}'.",
                provider=provider_id,
            )
        return factory_obj

    metadata_resolver: Optional[MetadataResolver] = None
    if metadata_attr:
        metadata_resolver = _build_attr_resolver(
            module_path, metadata_attr, provider_id
        )

    availability_check: Optional[AvailabilityCheck] = None
    if availability_attr:
        availability_check = _build_attr_resolver(
            module_path, availability_attr, provider_id
        )

    register_provider(
        provider_id,
        lazy_loader=_lazy_loader,
        metadata_resolver=metadata_resolver,
        availability_check=availability_check,
        priority=priority,
        description=description,
        tags=tags,
        replace=replace,
    )


def _build_attr_resolver(
    module_path: str, attr: str, provider_id: str
) -> Callable[[], Any]:
    """Build a lazy attribute resolver for a module."""

    def _resolver() -> Any:
        module = importlib.import_module(module_path)
        target = getattr(module, attr, None)
        if callable(target):
            return target()
        return target

    return _resolver


# =============================================================================
# Public API - Resolution
# =============================================================================


def available_providers(*, include_unavailable: bool = False) -> List[str]:
    """
    Return provider identifiers sorted by priority (desc) then name.

    Args:
        include_unavailable: If True, include providers that fail availability check

    Returns:
        List of provider IDs sorted by priority (descending), then alphabetically

    Example:
        >>> available_providers()
        ['gemini', 'codex', 'cursor-agent']
        >>> available_providers(include_unavailable=True)
        ['gemini', 'codex', 'cursor-agent', 'opencode']
    """
    providers: List[ProviderRegistration] = list(_REGISTRY.values())
    providers.sort(key=lambda reg: (-reg.priority, reg.provider_id))

    if include_unavailable:
        return [reg.provider_id for reg in providers]

    return [reg.provider_id for reg in providers if reg.is_available()]


def check_provider_available(provider_id: str) -> bool:
    """
    Check if a provider is available using its registered availability check.

    Args:
        provider_id: Provider identifier (e.g., "gemini", "codex")

    Returns:
        True if provider is registered and passes its availability check,
        False otherwise

    Example:
        >>> check_provider_available("gemini")
        True
        >>> check_provider_available("nonexistent")
        False
    """
    registration = _REGISTRY.get(provider_id)
    if registration is None:
        return False
    return registration.is_available()


def resolve_provider(
    provider_id: str,
    *,
    hooks: ProviderHooks,
    model: Optional[str] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> ProviderContext:
    """
    Instantiate a provider by ID using the registered factory.

    Args:
        provider_id: Provider identifier (e.g., "gemini", "codex")
        hooks: Lifecycle hooks to wire into the provider
        model: Optional model override
        overrides: Optional provider-specific configuration overrides

    Returns:
        Instantiated ProviderContext

    Raises:
        ProviderUnavailableError: If provider not registered or unavailable

    Example:
        >>> hooks = ProviderHooks()
        >>> provider = resolve_provider("gemini", hooks=hooks, model="pro")
        >>> result = provider.generate(request)
    """
    registration = _REGISTRY.get(provider_id)
    if registration is None:
        raise ProviderUnavailableError(
            f"Provider '{provider_id}' is not registered.", provider=provider_id
        )

    if not registration.is_available():
        raise ProviderUnavailableError(
            f"Provider '{provider_id}' is currently unavailable.",
            provider=provider_id,
        )

    factory = registration.load_factory()
    dependencies = _resolve_dependencies(provider_id)
    return factory(
        hooks=hooks,
        model=model,
        dependencies=dependencies,
        overrides=overrides,
    )


def get_provider_metadata(provider_id: str) -> Optional[ProviderMetadata]:
    """
    Return ProviderMetadata for a registered provider.

    Args:
        provider_id: Provider identifier

    Returns:
        ProviderMetadata if available, None otherwise
    """
    registration = _REGISTRY.get(provider_id)
    if registration is None:
        return None
    return registration.resolve_metadata()


def describe_providers() -> List[Dict[str, object]]:
    """
    Return descriptive information for all registered providers.

    Returns:
        List of dicts with provider info (id, description, priority, tags, available)

    Example:
        >>> describe_providers()
        [
            {
                "id": "gemini",
                "description": "Google Gemini CLI",
                "priority": 10,
                "tags": ["external", "cli"],
                "available": True,
            },
            ...
        ]
    """
    summary: List[Dict[str, object]] = []
    for reg in _REGISTRY.values():
        summary.append(
            {
                "id": reg.provider_id,
                "description": reg.description,
                "priority": reg.priority,
                "tags": list(reg.tags),
                "available": reg.is_available(),
            }
        )
    return summary


# =============================================================================
# Public API - Dependency Injection
# =============================================================================


def set_dependency_resolver(resolver: Optional[DependencyResolver]) -> None:
    """
    Register a callable that supplies dependency dictionaries per provider ID.

    The resolver is invoked during `resolve_provider` and should return a dict
    that will be passed to the provider factory via the `dependencies` keyword.

    Args:
        resolver: Callable taking provider_id, returning dependency dict.
            Pass None to clear the resolver.

    Example:
        >>> def my_resolver(provider_id: str) -> Dict[str, object]:
        ...     if provider_id == "gemini":
        ...         return {"api_client": my_api_client}
        ...     return {}
        >>> set_dependency_resolver(my_resolver)
    """
    global _dependency_resolver
    _dependency_resolver = resolver


def _resolve_dependencies(provider_id: str) -> Dict[str, object]:
    """Resolve dependencies for a provider using the configured resolver."""
    if _dependency_resolver is None:
        return {}
    try:
        dependencies = _dependency_resolver(provider_id)
        return dependencies or {}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Dependency resolver failed for provider '%s': %s",
            provider_id,
            exc,
        )
        return {}


# =============================================================================
# Public API - Testing Support
# =============================================================================


def reset_registry() -> None:
    """
    Clear the registry and dependency resolver.

    Primarily used by tests to restore a clean state.
    """
    _REGISTRY.clear()
    set_dependency_resolver(None)
    logger.debug("Registry cleared")


def get_registration(provider_id: str) -> Optional[ProviderRegistration]:
    """
    Get the registration record for a provider (for testing/introspection).

    Args:
        provider_id: Provider identifier

    Returns:
        ProviderRegistration if found, None otherwise
    """
    return _REGISTRY.get(provider_id)


__all__ = [
    # Types
    "ProviderFactory",
    "ProviderRegistration",
    "AvailabilityCheck",
    "MetadataResolver",
    "LazyFactoryLoader",
    "DependencyResolver",
    # Registration
    "register_provider",
    "register_lazy_provider",
    # Resolution
    "available_providers",
    "check_provider_available",
    "resolve_provider",
    "get_provider_metadata",
    "describe_providers",
    # Dependency Injection
    "set_dependency_resolver",
    # Testing
    "reset_registry",
    "get_registration",
]
