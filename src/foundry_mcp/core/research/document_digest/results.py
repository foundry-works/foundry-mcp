"""Digest result types and serialization utilities.

Contains DigestResult dataclass and functions for serializing/deserializing
DigestPayload objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from foundry_mcp.core.research.models.digest import DigestPayload


@dataclass
class DigestResult:
    """Result of a document digest operation.

    Contains the digest payload along with execution metadata for
    performance tracking and cache management.

    Attributes:
        payload: The generated DigestPayload, or None if digestion failed
            or content was ineligible.
        cache_hit: Whether this result was retrieved from cache.
        duration_ms: Time taken to generate the digest in milliseconds.
        skipped: Whether digestion was skipped (content ineligible).
        skip_reason: Reason for skipping if skipped is True.
        warnings: List of warnings generated during digestion.
        metadata: Observability metadata dict containing _digest_cache_hit flag.
    """

    payload: Optional[DigestPayload] = None
    cache_hit: bool = False
    duration_ms: float = 0.0
    skipped: bool = False
    skip_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize metadata with cache hit flag."""
        self.metadata["_digest_cache_hit"] = self.cache_hit

    @property
    def success(self) -> bool:
        """Check if digest generation was successful."""
        return self.payload is not None and not self.skipped

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return len(self.warnings) > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dict representation suitable for API responses.
        """
        return {
            "payload": self.payload.model_dump() if self.payload else None,
            "cache_hit": self.cache_hit,
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "warnings": self.warnings,
            "success": self.success,
            "metadata": self.metadata,
        }


def serialize_payload(payload: DigestPayload) -> str:
    """Serialize a DigestPayload to a JSON string.

    Produces a valid JSON string representation of the payload that can be
    stored in source.content or transmitted over the wire.

    The output is deterministic (sorted keys) for consistent hashing and
    comparison. Uses compact encoding (no extra whitespace) for efficiency.

    Args:
        payload: The DigestPayload instance to serialize.

    Returns:
        JSON string representation of the payload.

    Raises:
        ValueError: If payload is None or serialization fails.

    Examples:
        >>> json_str = serialize_payload(payload)
        >>> '\"version\": \"1.0\"' in json_str
        True
        >>> json.loads(json_str)  # Valid JSON
        {...}
    """
    if payload is None:
        raise ValueError("Cannot serialize None payload")

    try:
        # Use Pydantic's model_dump for proper serialization
        data = payload.model_dump(mode="json")
        # Serialize with sorted keys for determinism
        return json.dumps(data, sort_keys=True, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to serialize payload: {e}") from e


def deserialize_payload(json_str: str) -> DigestPayload:
    """Deserialize a JSON string to a DigestPayload.

    Parses the JSON string and validates it against the DigestPayload schema.
    All field constraints (lengths, patterns, ranges) are enforced.

    Args:
        json_str: JSON string to deserialize.

    Returns:
        Validated DigestPayload instance.

    Raises:
        ValueError: If json_str is empty or not valid JSON.
        ValidationError: If data doesn't conform to DigestPayload schema.

    Examples:
        >>> payload = deserialize_payload(json_str)
        >>> payload.version
        '1.0'
        >>> payload.content_type
        'digest/v1'
    """
    if not json_str or not json_str.strip():
        raise ValueError("Cannot deserialize empty string")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    # Pydantic validation happens here - raises ValidationError on failure
    return DigestPayload.model_validate(data)


def validate_payload_dict(data: dict[str, Any]) -> DigestPayload:
    """Validate a dictionary against the DigestPayload schema.

    Useful for validating data from sources other than JSON strings,
    such as YAML or programmatic construction.

    Args:
        data: Dictionary to validate.

    Returns:
        Validated DigestPayload instance.

    Raises:
        ValidationError: If data doesn't conform to DigestPayload schema.
        TypeError: If data is not a dictionary.

    Examples:
        >>> data = {"version": "1.0", "content_type": "digest/v1", ...}
        >>> payload = validate_payload_dict(data)
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    return DigestPayload.model_validate(data)
