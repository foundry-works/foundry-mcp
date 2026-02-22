"""
Feature flag descriptors for discovery metadata.

Provides the FeatureFlagDescriptor dataclass used across discovery
metadata modules (environment, LLM, provider) for informational
capability descriptions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FeatureFlagDescriptor:
    """
    Descriptor for a feature used in capability negotiation metadata.

    Used by discovery metadata modules to describe server capabilities
    in a structured format for client consumption.

    Attributes:
        name: Unique identifier
        description: Human-readable description
        state: Lifecycle state (experimental, beta, stable, deprecated)
        default_enabled: Whether enabled by default
        percentage_rollout: Rollout percentage (0-100)
        dependencies: List of other features this depends on
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
