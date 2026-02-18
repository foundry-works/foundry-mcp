"""Data models for content summarization.

Provides enums, dataclasses, and type aliases used across the summarization
sub-package.

Key Components:
    - SummarizationLevel: Enum defining compression levels (RAW to HEADLINE)
    - SummarizationResult: Dataclass for summarization output with metadata
    - SummarizationConfig: Configuration for summarization behavior
    - SummarizationFunc: Type alias for provider function signatures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from foundry_mcp.core.errors.research import (
    SummarizationValidationError,
)

from .constants import CHARS_PER_TOKEN, MAX_RETRIES, RETRY_DELAY, DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP


class SummarizationLevel(str, Enum):
    """Summarization compression levels.

    Defines how aggressively content should be summarized, from raw
    passthrough to extreme compression.

    Levels:
        RAW: No summarization, content passed through unchanged
        CONDENSED: Light compression, preserving most details (~50-70% of original)
        KEY_POINTS: Medium compression, extracting main points (~20-40% of original)
        HEADLINE: Extreme compression, single sentence or title (~5-10% of original)

    Example:
        level = SummarizationLevel.KEY_POINTS
        # Content: "This is a long article about machine learning. It covers
        #           neural networks, training methods, and applications..."
        # Summary: "- Neural networks overview - Training methodologies
        #           - Real-world applications"
    """

    RAW = "raw"
    CONDENSED = "condensed"
    KEY_POINTS = "key_points"
    HEADLINE = "headline"

    @property
    def target_compression_ratio(self) -> float:
        """Get the target compression ratio for this level.

        Returns:
            Approximate fraction of original content to retain (0.0-1.0)
        """
        return {
            SummarizationLevel.RAW: 1.0,
            SummarizationLevel.CONDENSED: 0.6,
            SummarizationLevel.KEY_POINTS: 0.3,
            SummarizationLevel.HEADLINE: 0.1,
        }[self]

    @property
    def max_output_tokens(self) -> int:
        """Get recommended max output tokens for this level.

        Returns:
            Suggested maximum tokens for summarized output
        """
        return {
            SummarizationLevel.RAW: 0,  # No limit (passthrough)
            SummarizationLevel.CONDENSED: 2000,
            SummarizationLevel.KEY_POINTS: 500,
            SummarizationLevel.HEADLINE: 100,
        }[self]

    def next_tighter_level(self) -> Optional["SummarizationLevel"]:
        """Get the next more aggressive summarization level.

        Returns:
            Next tighter level, or None if already at HEADLINE
        """
        progression = [
            SummarizationLevel.RAW,
            SummarizationLevel.CONDENSED,
            SummarizationLevel.KEY_POINTS,
            SummarizationLevel.HEADLINE,
        ]
        try:
            idx = progression.index(self)
            if idx < len(progression) - 1:
                return progression[idx + 1]
        except ValueError:
            pass
        return None


@dataclass
class SummarizationResult:
    """Result of a summarization operation.

    Contains the summarized content along with metadata about the
    summarization process. Supports per-level validation requirements.

    Attributes:
        content: The summarized text (required for all levels)
        level: Summarization level that was used
        key_points: List of extracted key points (required for KEY_POINTS level)
        source_ids: List of source identifiers for provenance tracking
        original_tokens: Estimated tokens in the original content
        summary_tokens: Estimated tokens in the summary
        provider_id: Provider that generated the summary (if known)
        truncated: Whether the result was truncated as a last resort
        warnings: List of warnings generated during summarization

    Level Requirements:
        - RAW: content only (passthrough)
        - CONDENSED: content required
        - KEY_POINTS: content + key_points required
        - HEADLINE: content only (single sentence)

    Example:
        result = SummarizationResult(
            content="Article discusses AI advances...",
            level=SummarizationLevel.KEY_POINTS,
            key_points=["AI making progress", "New models released"],
            source_ids=["article-123"],
        )
        result.validate()  # Raises if missing required fields
    """

    content: str
    level: SummarizationLevel
    key_points: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    original_tokens: int = 0
    summary_tokens: int = 0
    provider_id: Optional[str] = None
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Calculate the actual compression ratio achieved.

        Returns:
            Ratio of summary_tokens to original_tokens (0.0-1.0)
        """
        if self.original_tokens <= 0:
            return 1.0
        return self.summary_tokens / self.original_tokens

    def validate(self) -> bool:
        """Validate the result meets level-specific requirements.

        Returns:
            True if validation passes

        Raises:
            SummarizationValidationError: If required fields are missing
        """
        missing: list[str] = []

        # All levels require content
        if not self.content or not self.content.strip():
            missing.append("content")

        # KEY_POINTS level requires key_points list
        if self.level == SummarizationLevel.KEY_POINTS:
            if not self.key_points:
                missing.append("key_points")

        if missing:
            raise SummarizationValidationError(
                "Summarization result failed validation",
                self.level,
                missing,
            )

        return True

    def is_valid(self) -> bool:
        """Check if the result meets level-specific requirements.

        Unlike validate(), this returns False instead of raising.

        Returns:
            True if valid, False otherwise
        """
        try:
            return self.validate()
        except SummarizationValidationError:
            return False

    @classmethod
    def from_raw_output(
        cls,
        raw_output: str,
        level: SummarizationLevel,
        *,
        source_ids: Optional[list[str]] = None,
        original_tokens: int = 0,
        provider_id: Optional[str] = None,
    ) -> "SummarizationResult":
        """Parse raw LLM output into a SummarizationResult.

        Attempts to extract key_points from bullet-formatted output
        for KEY_POINTS level summarization.

        Args:
            raw_output: Raw text output from LLM
            level: Summarization level used
            source_ids: Source identifiers for provenance
            original_tokens: Original content token count
            provider_id: Provider that generated the output

        Returns:
            Parsed SummarizationResult
        """
        content = raw_output.strip()
        key_points: list[str] = []

        # For KEY_POINTS level, try to extract bullet points
        if level == SummarizationLevel.KEY_POINTS:
            key_points = cls._extract_key_points(content)

        return cls(
            content=content,
            level=level,
            key_points=key_points,
            source_ids=source_ids or [],
            original_tokens=original_tokens,
            summary_tokens=len(content) // 4,  # Estimate
            provider_id=provider_id,
        )

    @staticmethod
    def _extract_key_points(content: str) -> list[str]:
        """Extract bullet points from content.

        Looks for lines starting with -, *, or numbered bullets.

        Args:
            content: Text containing bullet points

        Returns:
            List of extracted key points
        """
        key_points = []
        for line in content.split("\n"):
            line = line.strip()
            # Check for bullet markers
            if line.startswith(("-", "*", "\u2022")):
                point = line.lstrip("-*\u2022 ").strip()
                if point:
                    key_points.append(point)
            # Check for numbered lists (1., 2., etc.)
            elif len(line) > 2 and line[0].isdigit() and line[1] in ".)" :
                point = line[2:].strip()
                if point:
                    key_points.append(point)

        return key_points

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dict representation of the result
        """
        return {
            "content": self.content,
            "level": self.level.value,
            "key_points": self.key_points,
            "source_ids": self.source_ids,
            "original_tokens": self.original_tokens,
            "summary_tokens": self.summary_tokens,
            "provider_id": self.provider_id,
            "truncated": self.truncated,
            "warnings": self.warnings,
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class SummarizationConfig:
    """Configuration for content summarization.

    Attributes:
        summarization_provider: Primary provider for summarization
        summarization_providers: Fallback providers (tried in order if primary fails)
        max_retries: Maximum retry attempts per provider
        retry_delay: Delay between retries in seconds
        timeout: Timeout per summarization request in seconds
        chunk_size: Maximum tokens per chunk for large content
        chunk_overlap: Token overlap between chunks
        target_budget: Target output token budget (triggers re-summarization if exceeded)
        cache_enabled: Whether to cache summarization results (default True)
    """

    summarization_provider: Optional[str] = None
    summarization_providers: list[str] = field(default_factory=list)
    max_retries: int = MAX_RETRIES
    retry_delay: float = RETRY_DELAY
    timeout: float = 60.0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    target_budget: Optional[int] = None  # None = no budget enforcement
    cache_enabled: bool = True  # Enable summary caching by default

    def get_provider_chain(self) -> list[str]:
        """Get ordered list of providers to try.

        Returns primary provider first, followed by fallback providers.
        Deduplicates the list while preserving order.

        Returns:
            Ordered list of provider IDs to try
        """
        chain = []
        seen = set()

        # Add primary provider first
        if self.summarization_provider:
            chain.append(self.summarization_provider)
            seen.add(self.summarization_provider)

        # Add fallback providers
        for provider in self.summarization_providers:
            if provider not in seen:
                chain.append(provider)
                seen.add(provider)

        return chain


# Type alias for the summarization function signature
SummarizationFunc = Callable[[str, SummarizationLevel, str], Any]
