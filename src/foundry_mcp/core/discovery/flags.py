"""
Feature flag descriptors and autonomy feature flags for discovery.

Provides the FeatureFlagDescriptor dataclass used across all discovery
metadata modules, plus autonomy-specific feature flags and capabilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


# Autonomy feature flags for capability negotiation
AUTONOMY_FEATURE_FLAGS: Dict[str, FeatureFlagDescriptor] = {
    "autonomy_sessions": FeatureFlagDescriptor(
        name="autonomy_sessions",
        description="Autonomous session management for continuous task execution with persistence",
        state="experimental",
        default_enabled=False,
        percentage_rollout=0,
        dependencies=[],
    ),
    "autonomy_fidelity_gates": FeatureFlagDescriptor(
        name="autonomy_fidelity_gates",
        description="Fidelity gates for autonomous execution - quality checkpoints between phases",
        state="experimental",
        default_enabled=False,
        percentage_rollout=0,
        dependencies=["autonomy_sessions"],
    ),
    "autonomy_gate_invariants": FeatureFlagDescriptor(
        name="autonomy_gate_invariants",
        description="Gate invariant observability - exposes required/satisfied/missing gates in API responses",
        state="beta",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["autonomy_sessions"],
    ),
}


def get_autonomy_capabilities() -> Dict[str, Any]:
    """
    Get autonomy-related capabilities for capability negotiation.

    Returns:
        Dict with autonomy feature flags and session management support.
    """
    return {
        "autonomy": {
            "supported": True,
            "description": "Autonomous execution with session persistence and fidelity gates",
            "actions": ["session", "session-step"],
            "tools": ["task"],
        },
        "gate_invariants": {
            "supported": True,
            "description": "Required/satisfied/missing gate exposure in session-step responses",
            "response_fields": ["required_phase_gates", "satisfied_gates", "missing_required_gates"],
            "config_fields": ["enforce_required_phase_gates", "allow_gate_waiver"],
        },
        "feature_flags": {
            name: flag.to_dict() for name, flag in AUTONOMY_FEATURE_FLAGS.items()
        },
    }


def is_autonomy_feature_flag(flag_name: str) -> bool:
    """
    Check if a flag name is an autonomy feature flag.

    Args:
        flag_name: Name of the flag to check

    Returns:
        True if flag is an autonomy feature flag
    """
    return flag_name in AUTONOMY_FEATURE_FLAGS


def get_autonomy_feature_flag(flag_name: str) -> Optional[FeatureFlagDescriptor]:
    """
    Get descriptor for a specific autonomy feature flag.

    Args:
        flag_name: Name of the autonomy feature flag

    Returns:
        FeatureFlagDescriptor if found, None otherwise
    """
    return AUTONOMY_FEATURE_FLAGS.get(flag_name)
