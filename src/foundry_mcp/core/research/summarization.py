"""Content summarization utilities for deep research workflows.

Provides LLM-based content compression with configurable summarization levels,
provider chain with fallback, retry logic, and caching support.

Key Components:
    - SummarizationLevel: Enum defining compression levels (RAW to HEADLINE)
    - ContentSummarizer: Main class for summarizing content with provider chain

Usage:
    from foundry_mcp.core.research.summarization import (
        ContentSummarizer,
        SummarizationLevel,
    )

    # Create summarizer with provider configuration
    summarizer = ContentSummarizer(
        summarization_provider="claude",
        summarization_providers=["gemini", "codex"],
    )

    # Summarize content
    result = await summarizer.summarize(
        content="Long article text...",
        level=SummarizationLevel.KEY_POINTS,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 3.0  # seconds

# Chunking configuration
DEFAULT_CHUNK_SIZE = 8000  # tokens (conservative for most models)
CHUNK_OVERLAP = 200  # tokens overlap between chunks
CHARS_PER_TOKEN = 4  # approximate for heuristic estimation

# Cache configuration
_SUMMARY_CACHE_MAX_SIZE = 1000  # Maximum cached summaries


class SummaryCache:
    """In-memory cache for summarization results.

    Caches summarization results using composite keys that include content hash,
    context hash, summarization level, and provider. This ensures cache
    invalidation when any relevant factor changes.

    The cache is bounded to prevent unbounded memory growth, using a simple
    half-flush eviction strategy when the limit is reached.

    Attributes:
        _cache: Internal dict mapping cache keys to SummarizationResult
        _enabled: Whether caching is enabled
        _max_size: Maximum number of entries

    Example:
        cache = SummaryCache(enabled=True)

        # Check cache before summarization
        result = cache.get(content, context, level, provider)
        if result is None:
            result = await summarizer._summarize_single(content, level, provider)
            cache.set(content, context, level, provider, result)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = _SUMMARY_CACHE_MAX_SIZE,
    ):
        """Initialize the summary cache.

        Args:
            enabled: Whether caching is enabled (default True)
            max_size: Maximum cache entries before eviction
        """
        self._cache: dict[tuple[str, str, str, str], "SummarizationResult"] = {}
        self._enabled = enabled
        self._max_size = max_size

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable caching."""
        self._enabled = value

    @staticmethod
    def _content_hash(content: str) -> str:
        """Generate a hash of content for cache keying.

        Uses SHA-256 truncated to 16 characters for reasonable uniqueness
        while keeping cache keys compact.

        Args:
            content: Text content to hash

        Returns:
            Hex string hash of the content
        """
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _make_key(
        self,
        content: str,
        context: Optional[str],
        level: "SummarizationLevel",
        provider_id: Optional[str],
    ) -> tuple[str, str, str, str]:
        """Create a cache key from the input parameters.

        Args:
            content: Content being summarized
            context: Optional context string
            level: Summarization level
            provider_id: Provider identifier

        Returns:
            Tuple of (content_hash, context_hash, level_value, provider_id)
        """
        content_hash = self._content_hash(content)
        context_hash = self._content_hash(context) if context else ""
        return (content_hash, context_hash, level.value, provider_id or "")

    def get(
        self,
        content: str,
        context: Optional[str],
        level: "SummarizationLevel",
        provider_id: Optional[str],
    ) -> Optional["SummarizationResult"]:
        """Retrieve a cached summarization result.

        Args:
            content: Content that was summarized
            context: Optional context string
            level: Summarization level
            provider_id: Provider identifier

        Returns:
            Cached SummarizationResult if found and cache enabled, None otherwise
        """
        if not self._enabled:
            return None

        key = self._make_key(content, context, level, provider_id)
        result = self._cache.get(key)

        if result is not None:
            logger.debug(f"Summary cache hit for {key[0][:8]}... at {level.value}")

        return result

    def set(
        self,
        content: str,
        context: Optional[str],
        level: "SummarizationLevel",
        provider_id: Optional[str],
        result: "SummarizationResult",
    ) -> None:
        """Store a summarization result in the cache.

        If the cache is full, evicts the oldest half of entries before adding.

        Args:
            content: Content that was summarized
            context: Optional context string
            level: Summarization level
            provider_id: Provider identifier
            result: The summarization result to cache
        """
        if not self._enabled:
            return

        # Evict oldest entries if at capacity (simple half-flush)
        if len(self._cache) >= self._max_size:
            keys_to_remove = list(self._cache.keys())[: self._max_size // 2]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"Summary cache evicted {len(keys_to_remove)} entries")

        key = self._make_key(content, context, level, provider_id)
        self._cache[key] = result
        logger.debug(f"Summary cache stored {key[0][:8]}... at {level.value}")

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries that were cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"Summary cache cleared {count} entries")
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with size, max_size, and enabled status
        """
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "enabled": self._enabled,
        }


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
        # Summary: "• Neural networks overview • Training methodologies
        #           • Real-world applications"
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


# Error classes (canonical definitions in foundry_mcp.core.errors.research)
from foundry_mcp.core.errors.research import (  # noqa: E402
    ProviderExhaustedError,
    SummarizationError,
    SummarizationValidationError,
)


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
            if line.startswith(("-", "*", "•")):
                point = line.lstrip("-*• ").strip()
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


class ContentSummarizer:
    """Content summarizer with provider chain and retry logic.

    Summarizes content using LLM providers with automatic fallback through
    a provider chain if the primary provider fails.

    Attributes:
        config: Summarization configuration
        _provider_func: Optional custom provider function for testing

    Example:
        summarizer = ContentSummarizer(
            summarization_provider="claude",
            summarization_providers=["gemini", "codex"],
        )

        # Summarize with automatic provider fallback
        result = await summarizer.summarize(
            content="Long text to summarize...",
            level=SummarizationLevel.KEY_POINTS,
        )
    """

    def __init__(
        self,
        summarization_provider: Optional[str] = None,
        summarization_providers: Optional[list[str]] = None,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY,
        timeout: float = 60.0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        target_budget: Optional[int] = None,
        cache_enabled: bool = True,
        *,
        provider_func: Optional[SummarizationFunc] = None,
    ):
        """Initialize the ContentSummarizer.

        Args:
            summarization_provider: Primary provider for summarization
            summarization_providers: Fallback providers (tried in order)
            max_retries: Maximum retry attempts per provider
            retry_delay: Delay between retries in seconds
            timeout: Timeout per summarization request in seconds
            chunk_size: Maximum tokens per chunk for large content
            chunk_overlap: Token overlap between chunks
            target_budget: Target output token budget (triggers re-summarization)
            cache_enabled: Whether to cache summarization results (default True)
            provider_func: Optional custom provider function (for testing)
        """
        self.config = SummarizationConfig(
            summarization_provider=summarization_provider,
            summarization_providers=summarization_providers or [],
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            target_budget=target_budget,
            cache_enabled=cache_enabled,
        )
        self._provider_func = provider_func
        self._cache = SummaryCache(enabled=cache_enabled)

    @classmethod
    def from_config(cls, config: SummarizationConfig) -> "ContentSummarizer":
        """Create summarizer from configuration object.

        Args:
            config: Summarization configuration

        Returns:
            Configured ContentSummarizer instance
        """
        return cls(
            summarization_provider=config.summarization_provider,
            summarization_providers=config.summarization_providers,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            timeout=config.timeout,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            target_budget=config.target_budget,
            cache_enabled=config.cache_enabled,
        )

    def get_provider_chain(self) -> list[str]:
        """Get the ordered list of providers to try.

        Returns:
            List of provider IDs in order of preference
        """
        return self.config.get_provider_chain()

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content using heuristic.

        Uses character-based approximation (4 chars per token).
        For more accurate counts, use the token_management module.

        Args:
            content: Text content

        Returns:
            Estimated token count
        """
        return max(1, len(content) // CHARS_PER_TOKEN)

    def _needs_chunking(self, content: str) -> bool:
        """Check if content exceeds chunk size and needs to be split.

        Args:
            content: Text content

        Returns:
            True if content needs chunking, False otherwise
        """
        return self._estimate_tokens(content) > self.config.chunk_size

    def _chunk_content(self, content: str) -> list[str]:
        """Split content into chunks with overlap.

        Splits on paragraph/sentence boundaries when possible to maintain
        coherence. Includes overlap between chunks to preserve context.

        Args:
            content: Text content to chunk

        Returns:
            List of content chunks
        """
        if not self._needs_chunking(content):
            return [content]

        # Convert token limits to character limits
        chunk_chars = self.config.chunk_size * CHARS_PER_TOKEN
        overlap_chars = self.config.chunk_overlap * CHARS_PER_TOKEN

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_chars

            # If this isn't the last chunk, try to break at a natural boundary
            if end < len(content):
                # Look for paragraph break in the last 20% of the chunk
                search_start = int(end * 0.8)
                para_break = content.rfind("\n\n", search_start, end)
                if para_break > start:
                    end = para_break

                # If no paragraph, look for sentence break
                elif (sentence_break := content.rfind(". ", search_start, end)) > start:
                    end = sentence_break + 1

            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, keeping some overlap for context
            # Advance by (chunk_size - overlap) to ensure progress
            step = chunk_chars - overlap_chars
            start = start + max(step, chunk_chars // 2)  # Ensure at least half chunk progress

        logger.debug(f"Split content into {len(chunks)} chunks")
        return chunks

    async def _summarize_single(
        self,
        content: str,
        level: SummarizationLevel,
        provider_id: Optional[str] = None,
    ) -> str:
        """Summarize a single chunk of content.

        This is the core summarization logic without chunking.

        Args:
            content: Content to summarize
            level: Summarization level
            provider_id: Override provider

        Returns:
            Summarized content

        Raises:
            ProviderExhaustedError: If all providers fail
        """
        # Handle RAW level (passthrough)
        if level == SummarizationLevel.RAW:
            return content

        # Determine provider chain
        if provider_id:
            chain = [provider_id]
        else:
            chain = self.get_provider_chain()

        if not chain:
            raise SummarizationError(
                "No summarization providers configured. Set summarization_provider "
                "or summarization_providers."
            )

        # Try each provider in chain
        errors: list[tuple[str, Exception]] = []

        for pid in chain:
            success, result, error = await self._try_provider_with_retries(
                pid, content, level
            )

            if success:
                return result

            if error:
                errors.append((pid, error))

        raise ProviderExhaustedError(errors)

    async def _map_reduce_summarize(
        self,
        chunks: list[str],
        level: SummarizationLevel,
        provider_id: Optional[str] = None,
    ) -> str:
        """Summarize multiple chunks using map-reduce pattern.

        Map phase: Summarize each chunk individually
        Reduce phase: Combine chunk summaries and summarize the combined result

        Args:
            chunks: List of content chunks
            level: Summarization level
            provider_id: Override provider

        Returns:
            Combined summary
        """
        logger.debug(f"Map-reduce summarization: {len(chunks)} chunks at {level.value}")

        # Map phase: summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Summarizing chunk {i + 1}/{len(chunks)}")
            summary = await self._summarize_single(chunk, level, provider_id)
            chunk_summaries.append(summary)

        # If only one chunk, return its summary directly
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Reduce phase: combine and re-summarize
        combined = "\n\n---\n\n".join(chunk_summaries)

        # If combined result still needs chunking, recurse
        if self._needs_chunking(combined):
            logger.debug("Combined summary still too large, recursing")
            return await self.summarize(combined, level, provider_id=provider_id)

        # Final reduction summary
        return await self._summarize_single(combined, level, provider_id)

    def _truncate_with_warning(
        self,
        content: str,
        max_tokens: int,
    ) -> str:
        """Truncate content to fit within token budget with warning.

        This is a last-resort fallback when summarization cannot meet
        the target budget.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated content with ellipsis indicator
        """
        max_chars = max_tokens * CHARS_PER_TOKEN
        if len(content) <= max_chars:
            return content

        logger.warning(
            f"Truncating summary from ~{self._estimate_tokens(content)} tokens "
            f"to {max_tokens} tokens (last resort)"
        )

        # Truncate and add ellipsis
        truncated = content[: max_chars - 20]  # Leave room for ellipsis

        # Try to break at sentence boundary
        last_period = truncated.rfind(". ")
        if last_period > max_chars // 2:
            truncated = truncated[: last_period + 1]

        return truncated + " [... truncated]"

    async def _call_provider(
        self,
        provider_id: str,
        content: str,
        level: SummarizationLevel,
    ) -> str:
        """Call a specific provider for summarization.

        Args:
            provider_id: Provider to use
            content: Content to summarize
            level: Summarization level

        Returns:
            Summarized content

        Raises:
            Exception: If provider call fails
        """
        if self._provider_func:
            # Use custom provider function (for testing)
            return await asyncio.to_thread(
                self._provider_func, content, level, provider_id
            )

        # Use real provider system
        from foundry_mcp.core.providers import (
            ProviderHooks,
            ProviderRequest,
            resolve_provider,
        )

        hooks = ProviderHooks()  # Default hooks (no-ops)
        provider = resolve_provider(provider_id, hooks=hooks)
        if provider is None:
            raise SummarizationError(f"Provider not available: {provider_id}")

        # Build summarization prompt
        prompt = self._build_prompt(content, level)

        provider_request = ProviderRequest(
            prompt=prompt,
            max_tokens=level.max_output_tokens or 2000,
            timeout=self.config.timeout,
        )

        # Run synchronous provider.generate in thread pool
        from foundry_mcp.core.providers import ProviderStatus

        result = await asyncio.to_thread(provider.generate, provider_request)
        if result.status != ProviderStatus.SUCCESS:
            error_msg = result.stderr or "Unknown error"
            raise SummarizationError(f"Provider {provider_id} failed: {error_msg}")

        return result.content

    def _build_prompt(self, content: str, level: SummarizationLevel) -> str:
        """Build the summarization prompt for the given level.

        Args:
            content: Content to summarize
            level: Summarization level

        Returns:
            Prompt string for the LLM
        """
        # Level-specific instructions
        instructions = {
            SummarizationLevel.RAW: "",
            SummarizationLevel.CONDENSED: (
                "Condense the following content while preserving key details and nuance. "
                "Target approximately 50-70% of the original length."
            ),
            SummarizationLevel.KEY_POINTS: (
                "Extract the key points from the following content as a concise bullet list. "
                "Focus on main ideas, findings, and conclusions. "
                "Target approximately 20-40% of the original length."
            ),
            SummarizationLevel.HEADLINE: (
                "Summarize the following content in a single sentence or brief headline. "
                "Capture the essential message in 1-2 lines maximum."
            ),
        }

        instruction = instructions.get(level, instructions[SummarizationLevel.KEY_POINTS])

        if level == SummarizationLevel.RAW:
            return content

        return f"{instruction}\n\nContent:\n{content}"

    async def _try_provider_with_retries(
        self,
        provider_id: str,
        content: str,
        level: SummarizationLevel,
    ) -> tuple[bool, str, Optional[Exception]]:
        """Try a provider with retry logic.

        Args:
            provider_id: Provider to try
            content: Content to summarize
            level: Summarization level

        Returns:
            Tuple of (success, result_or_empty, last_error)
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._call_provider(provider_id, content, level)
                logger.debug(
                    f"Summarization succeeded with {provider_id} "
                    f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                )
                return True, result, None

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Summarization attempt {attempt + 1} failed with {provider_id}: {e}"
                )

                # Don't retry on the last attempt
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)

        return False, "", last_error

    async def summarize(
        self,
        content: str,
        level: SummarizationLevel = SummarizationLevel.KEY_POINTS,
        *,
        provider_id: Optional[str] = None,
        target_budget: Optional[int] = None,
    ) -> str:
        """Summarize content using the provider chain with chunking support.

        Handles large content by splitting into chunks and using map-reduce.
        If the result exceeds the target budget, re-summarizes at tighter
        levels. Truncates as a last resort.

        Args:
            content: Content to summarize
            level: Summarization level (default: KEY_POINTS)
            provider_id: Override provider (skips chain logic if specified)
            target_budget: Target output token budget (overrides config)

        Returns:
            Summarized content

        Raises:
            ProviderExhaustedError: If all providers fail
            SummarizationError: If no providers are configured
        """
        # Handle RAW level (passthrough)
        if level == SummarizationLevel.RAW:
            return content

        # Determine effective budget
        budget = target_budget or self.config.target_budget

        # Check if content needs chunking
        if self._needs_chunking(content):
            logger.debug(
                f"Content exceeds chunk size ({self._estimate_tokens(content)} > "
                f"{self.config.chunk_size} tokens), using map-reduce"
            )
            chunks = self._chunk_content(content)
            result = await self._map_reduce_summarize(chunks, level, provider_id)
        else:
            # Single chunk - direct summarization
            result = await self._summarize_single(content, level, provider_id)

        # Post-check: enforce budget if specified
        if budget is not None:
            result = await self._enforce_budget(
                result, level, budget, provider_id
            )

        return result

    async def _enforce_budget(
        self,
        content: str,
        current_level: SummarizationLevel,
        target_budget: int,
        provider_id: Optional[str] = None,
    ) -> str:
        """Enforce token budget on summarized content.

        If content exceeds budget, steps down to more aggressive summarization
        levels. Truncates as a last resort.

        Args:
            content: Summarized content to check
            current_level: Current summarization level
            target_budget: Target token budget
            provider_id: Override provider

        Returns:
            Content within budget
        """
        estimated = self._estimate_tokens(content)

        # If within budget, return as-is
        if estimated <= target_budget:
            return content

        logger.debug(
            f"Summary exceeds budget ({estimated} > {target_budget} tokens), "
            f"trying tighter level"
        )

        # Try stepping down to tighter levels
        level = current_level
        while level is not None:
            next_level = level.next_tighter_level()
            if next_level is None:
                break

            level = next_level
            logger.debug(f"Re-summarizing at {level.value} level")

            try:
                result = await self._summarize_single(content, level, provider_id)
                estimated = self._estimate_tokens(result)

                if estimated <= target_budget:
                    return result

                # Update content for next iteration
                content = result

            except Exception as e:
                logger.warning(f"Re-summarization at {level.value} failed: {e}")
                break

        # Last resort: truncate with warning
        return self._truncate_with_warning(content, target_budget)

    def is_available(self) -> bool:
        """Check if at least one summarization provider is configured.

        Returns:
            True if providers are configured, False otherwise
        """
        return bool(self.get_provider_chain())

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache size, max_size, and enabled status
        """
        return self._cache.get_stats()

    def clear_cache(self) -> int:
        """Clear all cached summarization results.

        Returns:
            Number of entries that were cleared
        """
        return self._cache.clear()

    @property
    def cache_enabled(self) -> bool:
        """Check if summarization caching is enabled."""
        return self._cache.enabled

    @cache_enabled.setter
    def cache_enabled(self, value: bool) -> None:
        """Enable or disable summarization caching."""
        self._cache.enabled = value
        self.config.cache_enabled = value

    async def summarize_with_result(
        self,
        content: str,
        level: SummarizationLevel = SummarizationLevel.KEY_POINTS,
        *,
        provider_id: Optional[str] = None,
        target_budget: Optional[int] = None,
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> SummarizationResult:
        """Summarize content and return a detailed result object.

        Like summarize(), but returns a SummarizationResult with metadata
        instead of just the content string. Supports caching of results.

        Args:
            content: Content to summarize
            level: Summarization level (default: KEY_POINTS)
            provider_id: Override provider
            target_budget: Target output token budget
            context: Optional context string (affects cache key)
            use_cache: Whether to use cache for this request (default True)

        Returns:
            SummarizationResult with content and metadata
        """
        # Determine effective provider for cache key
        effective_provider = provider_id or self.config.summarization_provider

        # Check cache first (if enabled and requested)
        if use_cache:
            cached = self._cache.get(content, context, level, effective_provider)
            if cached is not None:
                return cached

        original_tokens = self._estimate_tokens(content)
        warnings: list[str] = []
        truncated = False

        # Perform summarization
        summary = await self.summarize(
            content, level, provider_id=provider_id, target_budget=target_budget
        )

        # Check if truncation occurred
        if "[... truncated]" in summary:
            truncated = True
            warnings.append("Content was truncated to fit budget")

        summary_tokens = self._estimate_tokens(summary)

        result = SummarizationResult(
            content=summary,
            level=level,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            provider_id=effective_provider,
            truncated=truncated,
            warnings=warnings,
        )

        # Store in cache (if enabled and requested)
        if use_cache:
            self._cache.set(content, context, level, effective_provider, result)

        return result

    async def batch_summarize(
        self,
        items: list[str],
        level: SummarizationLevel = SummarizationLevel.KEY_POINTS,
        *,
        provider_id: Optional[str] = None,
        total_budget: Optional[int] = None,
        per_item_budget: Optional[int] = None,
    ) -> list[SummarizationResult]:
        """Summarize multiple items efficiently with budget management.

        Processes items sequentially, respecting either a total budget
        across all items or a per-item budget.

        Budget allocation strategy:
        - If total_budget is set: Divides budget across items, with tighter
          summarization for later items if earlier ones use more than their share
        - If per_item_budget is set: Each item gets the same budget
        - If neither is set: No budget enforcement

        Args:
            items: List of content strings to summarize
            level: Summarization level for all items (default: KEY_POINTS)
            provider_id: Override provider for all items
            total_budget: Total token budget across all items
            per_item_budget: Budget per individual item

        Returns:
            List of SummarizationResult, one per input item

        Example:
            results = await summarizer.batch_summarize(
                items=["Article 1...", "Article 2...", "Article 3..."],
                level=SummarizationLevel.KEY_POINTS,
                total_budget=1000,
            )
            for r in results:
                print(f"Compressed {r.original_tokens} -> {r.summary_tokens} tokens")
        """
        if not items:
            return []

        results: list[SummarizationResult] = []
        remaining_budget = total_budget
        remaining_items = len(items)

        for i, item in enumerate(items):
            # Calculate budget for this item
            if per_item_budget is not None:
                item_budget = per_item_budget
            elif remaining_budget is not None and remaining_items > 0:
                # Allocate remaining budget evenly across remaining items
                item_budget = remaining_budget // remaining_items
            else:
                item_budget = None

            logger.debug(
                f"Batch item {i + 1}/{len(items)}: "
                f"budget={item_budget}, remaining_total={remaining_budget}"
            )

            try:
                result = await self.summarize_with_result(
                    item,
                    level,
                    provider_id=provider_id,
                    target_budget=item_budget,
                )
                results.append(result)

                # Update remaining budget
                if remaining_budget is not None:
                    remaining_budget = max(0, remaining_budget - result.summary_tokens)
                remaining_items -= 1

            except Exception as e:
                logger.error(f"Batch item {i + 1} failed: {e}")
                # Create error result
                results.append(
                    SummarizationResult(
                        content="",
                        level=level,
                        original_tokens=self._estimate_tokens(item),
                        summary_tokens=0,
                        truncated=False,
                        warnings=[f"Summarization failed: {e}"],
                    )
                )
                remaining_items -= 1

        return results

    async def batch_summarize_parallel(
        self,
        items: list[str],
        level: SummarizationLevel = SummarizationLevel.KEY_POINTS,
        *,
        provider_id: Optional[str] = None,
        per_item_budget: Optional[int] = None,
        max_concurrent: int = 3,
    ) -> list[SummarizationResult]:
        """Summarize multiple items in parallel with concurrency limit.

        Processes items concurrently for better performance. Note that
        total_budget cannot be used with parallel processing since items
        are processed simultaneously.

        Args:
            items: List of content strings to summarize
            level: Summarization level for all items
            provider_id: Override provider for all items
            per_item_budget: Budget per individual item
            max_concurrent: Maximum concurrent summarizations

        Returns:
            List of SummarizationResult in the same order as input items
        """
        if not items:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_item(item: str, index: int) -> tuple[int, SummarizationResult]:
            async with semaphore:
                try:
                    result = await self.summarize_with_result(
                        item,
                        level,
                        provider_id=provider_id,
                        target_budget=per_item_budget,
                    )
                    return index, result
                except Exception as e:
                    logger.error(f"Parallel batch item {index + 1} failed: {e}")
                    return index, SummarizationResult(
                        content="",
                        level=level,
                        original_tokens=self._estimate_tokens(item),
                        summary_tokens=0,
                        truncated=False,
                        warnings=[f"Summarization failed: {e}"],
                    )

        # Process all items concurrently
        tasks = [process_item(item, i) for i, item in enumerate(items)]
        indexed_results = await asyncio.gather(*tasks)

        # Sort by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        return [result for _, result in indexed_results]
