"""Digest configuration types.

Contains DigestPolicy enum and DigestConfig dataclass for controlling
document digest generation behavior.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

from foundry_mcp.core.research.models.sources import SourceQuality

logger = logging.getLogger(__name__)


class DigestPolicy(str, Enum):
    """Policy for when to apply digest compression.

    Controls whether and when sources are eligible for digest generation.

    Policies:
        OFF: Never digest - all sources pass through unchanged.
            Use when you want to preserve original content.
        AUTO: Automatic eligibility based on size and quality thresholds.
            Only HIGH and MEDIUM quality sources above size threshold are digested.
            This is the recommended default for most workflows.
        ALWAYS: Always digest sources that have content, regardless of
            size or quality. Use for aggressive compression scenarios.
    """

    OFF = "off"
    AUTO = "auto"
    ALWAYS = "always"


@dataclass
class DigestConfig:
    """Configuration for document digest generation.

    Attributes:
        policy: Digest eligibility policy (off/auto/always). Default is AUTO.
        min_content_length: Minimum content length (chars) to be eligible for digest.
            Content shorter than this is passed through unchanged. Only applies
            when policy is AUTO.
        quality_threshold: Minimum quality for auto policy. Sources must be
            this quality or higher to be eligible. Default is MEDIUM.
        max_summary_length: Maximum length of the summary field in DigestPayload.
        max_key_points: Maximum number of key points to extract.
        max_evidence_snippets: Maximum number of evidence snippets to include.
        max_snippet_length: Maximum length of each evidence snippet.
        include_evidence: Whether to include evidence snippets in digest output.
        chunk_size: Size of chunks for evidence extraction (in characters).
        chunk_overlap: Overlap between chunks for context preservation.
        cache_enabled: Whether to enable digest caching.
    """

    policy: DigestPolicy = DigestPolicy.AUTO
    min_content_length: int = 500
    quality_threshold: SourceQuality = SourceQuality.MEDIUM
    max_summary_length: int = 2000
    max_key_points: int = 10
    max_evidence_snippets: int = 10
    max_snippet_length: int = 500
    include_evidence: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 100
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        """Normalize config values to satisfy payload constraints."""
        # DigestPayload.summary max_length is 2000; clamp to prevent validation errors.
        if self.max_summary_length > 2000:
            logger.warning(
                "DigestConfig.max_summary_length=%d exceeds 2000; clamping to 2000",
                self.max_summary_length,
            )
            self.max_summary_length = 2000

        # DigestPayload.key_points max_length is 10 items; clamp to prevent validation errors.
        if self.max_key_points > 10:
            logger.warning(
                "DigestConfig.max_key_points=%d exceeds 10; clamping to 10",
                self.max_key_points,
            )
            self.max_key_points = 10

        # DigestPayload.evidence_snippets max_length is 10 items; clamp to prevent validation errors.
        if self.max_evidence_snippets > 10:
            logger.warning(
                "DigestConfig.max_evidence_snippets=%d exceeds 10; clamping to 10",
                self.max_evidence_snippets,
            )
            self.max_evidence_snippets = 10

        # EvidenceSnippet.text max_length is 500; clamp to prevent validation errors.
        if self.max_snippet_length > 500:
            logger.warning(
                "DigestConfig.max_snippet_length=%d exceeds 500; clamping to 500",
                self.max_snippet_length,
            )
            self.max_snippet_length = 500

    def compute_config_hash(self) -> str:
        """Compute a deterministic hash of configuration fields.

        Creates a hash from all configuration fields that affect digest
        output. Used for cache key generation to ensure cache invalidation
        when configuration changes.

        Fields included in hash (in order):
        - policy (digest policy)
        - min_content_length (min_chars threshold)
        - max_evidence_snippets (max sources)
        - include_evidence (whether evidence is included)
        - max_snippet_length (evidence_max_chars)
        - max_summary_length
        - max_key_points
        - chunk_size
        - chunk_overlap

        Returns:
            16-character lowercase hex hash string.

        Examples:
            >>> config = DigestConfig()
            >>> hash1 = config.compute_config_hash()
            >>> len(hash1)
            16
            >>> config2 = DigestConfig(max_evidence_snippets=5)
            >>> config.compute_config_hash() != config2.compute_config_hash()
            True
        """
        # Build tuple of all fields affecting digest output
        # Order matters for determinism
        config_tuple = (
            self.policy.value,  # digest policy
            self.min_content_length,  # min_chars
            self.max_evidence_snippets,  # max_sources
            self.include_evidence,  # include_evidence flag
            self.max_snippet_length,  # evidence_max_chars
            self.max_summary_length,
            self.max_key_points,
            self.chunk_size,
            self.chunk_overlap,
        )

        # Create deterministic string representation
        config_str = str(config_tuple)

        # Hash and truncate to 16 chars
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
