"""Document digest sub-package.

Split from monolithic document_digest.py for maintainability.
All public symbols re-exported for backward compatibility.
"""

from foundry_mcp.core.research.models.digest import DigestPayload, EvidenceSnippet

from .cache import DigestCache
from .config import DigestConfig, DigestPolicy
from .digestor import DIGEST_IMPL_VERSION, DocumentDigestor
from .results import (
    DigestResult,
    deserialize_payload,
    serialize_payload,
    validate_payload_dict,
)

__all__ = [
    # Configuration
    "DigestConfig",
    "DigestPolicy",
    # Caching
    "DigestCache",
    # Results & serialization
    "DigestResult",
    "serialize_payload",
    "deserialize_payload",
    "validate_payload_dict",
    # Core
    "DocumentDigestor",
    # Constants
    "DIGEST_IMPL_VERSION",
    # Re-exported from models for backward compatibility
    "DigestPayload",
    "EvidenceSnippet",
]
