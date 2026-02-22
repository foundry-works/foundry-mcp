"""
Server capabilities and negotiation for MCP clients.

Provides ServerCapabilities, capability negotiation, and the
get_capabilities() handshake endpoint.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import SCHEMA_VERSION


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
        autonomy_sessions: Whether autonomous session management is supported
        autonomy_fidelity_gates: Whether fidelity gates for autonomous execution are enabled
        autonomy_gate_invariants: Whether gate invariant observability is exposed in responses
        gate_enforcement_default: Default gate enforcement mode (strict, lenient, disabled)
    """

    response_version: str = "response-v2"
    supports_streaming: bool = False
    supports_batch: bool = True
    supports_pagination: bool = True
    max_batch_size: int = 100
    rate_limit_headers: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ["json"])
    autonomy_sessions: bool = True
    autonomy_fidelity_gates: bool = True
    autonomy_gate_invariants: bool = True
    gate_enforcement_default: str = "strict"

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
            "autonomy_sessions": self.autonomy_sessions,
            "autonomy_fidelity_gates": self.autonomy_fidelity_gates,
            "autonomy_gate_invariants": self.autonomy_gate_invariants,
            "gate_enforcement_default": self.gate_enforcement_default,
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

    capabilities = _capabilities.to_dict()

    autonomy_enabled = {
        "autonomy_sessions": True,
        "autonomy_fidelity_gates": True,
        "autonomy_gate_invariants": True,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "capabilities": capabilities,
        "runtime": {
            "autonomy": {
                "supported_by_binary": autonomy_enabled,
                "enabled_now": autonomy_enabled,
            },
            "conventions": {
                "discovery_as_hints": True,
                "responses_as_truth": True,
                "description": (
                    "Manifest/discovery metadata describes support; runtime action "
                    "responses and capability payloads report currently enabled state."
                ),
            },
        },
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
                f"Requested version '{requested_version}' not supported, using '{_capabilities.response_version}'"
            )

    # Feature negotiation
    available_features = {
        "streaming": _capabilities.supports_streaming,
        "batch": _capabilities.supports_batch,
        "pagination": _capabilities.supports_pagination,
        "rate_limit_headers": _capabilities.rate_limit_headers,
        "autonomy_sessions": _capabilities.autonomy_sessions,
        "autonomy_fidelity_gates": _capabilities.autonomy_fidelity_gates,
        "autonomy_gate_invariants": _capabilities.autonomy_gate_invariants,
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
