"""Document digest generation for deep research workflows.

Provides the DocumentDigestor class for compressing source content into
structured digests (DigestPayload) with evidence snippets and citation
traceability.

Key Components:
    - DigestResult: Dataclass containing digest payload and execution metadata
    - DigestConfig: Configuration for digest generation
    - DocumentDigestor: Main class for generating digests from source content

Usage:
    from foundry_mcp.core.research.document_digest import (
        DocumentDigestor,
        DigestConfig,
        DigestResult,
    )
    from foundry_mcp.core.research.summarization import ContentSummarizer
    from foundry_mcp.core.research.pdf_extractor import PDFExtractor

    # Create dependencies
    summarizer = ContentSummarizer(summarization_provider="claude")
    pdf_extractor = PDFExtractor()
    config = DigestConfig()

    # Create digestor
    digestor = DocumentDigestor(
        summarizer=summarizer,
        pdf_extractor=pdf_extractor,
        config=config,
    )

    # Generate digest
    result = await digestor.digest(content, query="research query")
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.research.models import DigestPayload, EvidenceSnippet, SourceQuality
from foundry_mcp.core.research.pdf_extractor import PDFExtractor
from foundry_mcp.core.research.summarization import (
    ContentSummarizer,
    SummarizationLevel,
)

# Initialize metrics collector
_metrics = get_metrics()


# =============================================================================
# Enums
# =============================================================================


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

logger = logging.getLogger(__name__)


# =============================================================================
# Version Constants
# =============================================================================

# Digest implementation version. Bump when algorithm changes to invalidate caches.
DIGEST_IMPL_VERSION = "1.0"


# =============================================================================
# Configuration
# =============================================================================


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


# =============================================================================
# Digest Cache
# =============================================================================

# Default maximum cache size
_DIGEST_CACHE_MAX_SIZE = 100


class DigestCache:
    """In-memory cache for digest results.

    Caches DigestResult objects using composite keys that include source ID,
    content hash, query hash, and config hash. This ensures cache invalidation
    when any relevant factor changes.

    The cache is bounded to prevent unbounded memory growth, using a simple
    half-flush eviction strategy when the limit is reached.

    Attributes:
        _cache: Internal dict mapping cache keys to DigestResult
        _enabled: Whether caching is enabled
        _max_size: Maximum number of entries

    Example:
        cache = DigestCache(enabled=True)

        # Check cache before digestion
        result = cache.get(cache_key)
        if result is None:
            result = await digestor._generate_digest(...)
            cache.set(cache_key, result)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_size: int = _DIGEST_CACHE_MAX_SIZE,
    ):
        """Initialize the digest cache.

        Args:
            enabled: Whether caching is enabled (default True)
            max_size: Maximum cache entries before eviction
        """
        self._cache: dict[str, "DigestResult"] = {}
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

    def get(self, cache_key: str) -> Optional["DigestResult"]:
        """Retrieve a cached digest result.

        Args:
            cache_key: Cache key from generate_cache_key()

        Returns:
            Cached DigestResult if found and cache enabled, None otherwise
        """
        if not self._enabled:
            return None

        result = self._cache.get(cache_key)

        if result is not None:
            logger.debug(f"Digest cache hit for key {cache_key[:30]}...")

        return result

    def set(self, cache_key: str, result: "DigestResult") -> None:
        """Store a digest result in the cache.

        If the cache is full, performs half-flush eviction (removes oldest
        half of entries) before storing the new result.

        Args:
            cache_key: Cache key from generate_cache_key()
            result: DigestResult to cache
        """
        if not self._enabled:
            return

        # Evict if at capacity (half-flush strategy)
        if len(self._cache) >= self._max_size:
            keys = list(self._cache.keys())
            for key in keys[:len(keys) // 2]:
                del self._cache[key]
            logger.debug(f"Digest cache eviction: removed {len(keys) // 2} entries")

        self._cache[cache_key] = result
        logger.debug(f"Digest cached for key {cache_key[:30]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


# =============================================================================
# Result Dataclass
# =============================================================================


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


# =============================================================================
# DocumentDigestor Class
# =============================================================================


class DocumentDigestor:
    """Generates structured digests from document content.

    The DocumentDigestor compresses source content into DigestPayload objects
    containing summaries, key points, and evidence snippets with citation
    locators. It uses the ContentSummarizer for text compression and
    PDFExtractor for handling PDF documents.

    The digestion process:
    1. Check eligibility (content length, type)
    2. Normalize text to canonical form
    3. Generate summary and key points via summarizer
    4. Extract evidence snippets with relevance scoring
    5. Compute content hash for archival linkage
    6. Package into DigestPayload

    Attributes:
        summarizer: ContentSummarizer instance for text summarization.
        pdf_extractor: PDFExtractor instance for PDF text extraction.
        config: DigestConfig with generation parameters.

    Example:
        summarizer = ContentSummarizer(summarization_provider="claude")
        pdf_extractor = PDFExtractor()
        config = DigestConfig(min_content_length=1000)

        digestor = DocumentDigestor(
            summarizer=summarizer,
            pdf_extractor=pdf_extractor,
            config=config,
        )

        # Digest text content
        result = await digestor.digest(
            content="Long article text...",
            query="What are the key findings?",
        )

        if result.success:
            print(f"Summary: {result.payload.summary}")
            print(f"Key points: {result.payload.key_points}")
    """

    def __init__(
        self,
        summarizer: ContentSummarizer,
        pdf_extractor: PDFExtractor,
        config: Optional[DigestConfig] = None,
        cache: Optional[DigestCache] = None,
    ) -> None:
        """Initialize DocumentDigestor with dependencies.

        Args:
            summarizer: ContentSummarizer instance for generating summaries
                and key points from content.
            pdf_extractor: PDFExtractor instance for extracting text from
                PDF documents with page boundary tracking.
            config: Optional DigestConfig for customizing digest generation.
                If not provided, uses default configuration.
            cache: Optional DigestCache for caching digest results.
                If not provided and caching is enabled, creates a new cache.
        """
        self.summarizer = summarizer
        self.pdf_extractor = pdf_extractor
        self.config = config or DigestConfig()

        # Initialize cache based on config
        if cache is not None:
            self._cache = cache
        else:
            self._cache = DigestCache(enabled=self.config.cache_enabled)

        # Circuit breaker state for tracking attempts in a sliding window
        # Each entry is (timestamp, success_bool)
        self._attempt_window: list[tuple[float, bool]] = []
        self._window_size = 10  # Number of recent operations to track
        self._failure_threshold_ratio = 0.7  # 70% failure rate triggers breaker
        self._min_samples = 5  # Minimum samples before ratio applies
        self._circuit_breaker_open = False
        self._circuit_breaker_opened_at: Optional[float] = None
        self._circuit_breaker_reset_seconds = 60.0  # Auto-reset after 60 seconds

        # Legacy attributes for backward compatibility with existing code
        self._failure_window: list[float] = []  # Deprecated, use _attempt_window
        self._failure_window_size = self._window_size
        self._failure_threshold = int(self._window_size * self._failure_threshold_ratio)
        self._circuit_breaker_triggered = False  # Alias for _circuit_breaker_open

        logger.debug(
            f"DocumentDigestor initialized with config: "
            f"min_content_length={self.config.min_content_length}, "
            f"cache_enabled={self.config.cache_enabled}"
        )

    def _record_attempt(self, success: bool) -> None:
        """Record a digest attempt (success or failure) for circuit breaker.

        Maintains a sliding window of recent attempts. When failure ratio exceeds
        70% with at least 5 samples, the circuit breaker opens.

        Args:
            success: Whether the attempt was successful.
        """
        now = time.time()
        self._attempt_window.append((now, success))

        # Trim window to max size (keep most recent)
        if len(self._attempt_window) > self._window_size:
            self._attempt_window = self._attempt_window[-self._window_size:]

        # Calculate failure ratio
        total_attempts = len(self._attempt_window)
        failures = sum(1 for _, s in self._attempt_window if not s)
        failure_ratio = failures / total_attempts if total_attempts > 0 else 0.0

        # Check if threshold exceeded (only with minimum samples)
        if (
            total_attempts >= self._min_samples
            and failure_ratio >= self._failure_threshold_ratio
            and not self._circuit_breaker_open
        ):
            self._circuit_breaker_open = True
            self._circuit_breaker_opened_at = now
            self._circuit_breaker_triggered = True  # Legacy alias
            audit_log(
                "digest.circuit_breaker_triggered",
                window_failures=failures,
                window_size=total_attempts,
                failure_ratio=round(failure_ratio, 2),
                failure_threshold=self._failure_threshold_ratio,
            )
            logger.warning(
                "Digest circuit breaker opened: %.0f%% failures (%d/%d) in window",
                failure_ratio * 100,
                failures,
                total_attempts,
            )

    def _record_failure(self) -> None:
        """Record a digest failure and check for circuit breaker triggering.

        Maintains a sliding window of attempts. When failure ratio exceeds
        70% with at least 5 samples, emits a digest.circuit_breaker_triggered
        audit event.
        """
        self._record_attempt(success=False)
        # Legacy: also append to old failure_window for backward compatibility
        self._failure_window.append(time.time())
        if len(self._failure_window) > self._failure_window_size:
            self._failure_window = self._failure_window[-self._failure_window_size:]

    def _record_success(self) -> None:
        """Record a successful digest operation.

        Records success in the attempt window. If failure ratio drops below
        threshold, the circuit breaker closes.
        """
        self._record_attempt(success=True)

        # Check if circuit breaker should close (ratio dropped below threshold)
        total_attempts = len(self._attempt_window)
        failures = sum(1 for _, s in self._attempt_window if not s)
        failure_ratio = failures / total_attempts if total_attempts > 0 else 0.0

        if self._circuit_breaker_open and failure_ratio < self._failure_threshold_ratio:
            self._circuit_breaker_open = False
            self._circuit_breaker_opened_at = None
            self._circuit_breaker_triggered = False  # Legacy alias
            logger.info(
                "Digest circuit breaker closed: %.0f%% failures (%d/%d) - below threshold",
                failure_ratio * 100,
                failures,
                total_attempts,
            )

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (should skip digest attempts).

        The circuit breaker auto-resets after 60 seconds to allow retry.

        Returns:
            True if circuit breaker is open and digest should be skipped.
        """
        if not self._circuit_breaker_open:
            return False

        # Check for auto-reset after timeout
        if self._circuit_breaker_opened_at is not None:
            elapsed = time.time() - self._circuit_breaker_opened_at
            if elapsed >= self._circuit_breaker_reset_seconds:
                logger.info(
                    "Digest circuit breaker auto-reset after %.1f seconds",
                    elapsed,
                )
                self._circuit_breaker_open = False
                self._circuit_breaker_opened_at = None
                self._circuit_breaker_triggered = False
                # Clear attempt window to start fresh
                self._attempt_window.clear()
                return False

        return True

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (e.g., for new iteration).

        Call this at the start of a new research iteration to allow
        retrying digests even if the breaker was previously open.
        """
        self._circuit_breaker_open = False
        self._circuit_breaker_opened_at = None
        self._circuit_breaker_triggered = False
        self._attempt_window.clear()
        self._failure_window.clear()
        logger.debug("Digest circuit breaker manually reset")

    async def digest(
        self,
        source: str,
        query: str,
        *,
        source_id: Optional[str] = None,
        quality: Optional[SourceQuality] = None,
        page_boundaries: Optional[list[tuple[int, int, int]]] = None,
    ) -> DigestResult:
        """Generate a structured digest from source content.

        Compresses source content into a DigestPayload containing a summary,
        key points, and evidence snippets. The digest is query-conditioned,
        meaning the summary focus and evidence selection depend on the
        research query provided.

        Args:
            source: The source content to digest (text string).
            query: The research query to condition the digest on.
                Used for focusing the summary and selecting relevant evidence.
            source_id: Optional source identifier for cache keying.
                If provided and caching is enabled, results may be cached.
            quality: Optional source quality level for eligibility filtering.
                When policy is AUTO, only HIGH and MEDIUM quality sources
                are eligible for digestion.
            page_boundaries: Optional list of PDF page boundaries in the source
                text. Each entry is (page_number, start_offset, end_offset) using
                0-based offsets into the raw source text. When provided, digest
                locators include page numbers (page:N:char:S-E).

        Returns:
            DigestResult containing the DigestPayload and execution metadata.
            If content is ineligible (policy, size, or quality), returns a
            result with skipped=True and no payload.

        Example:
            result = await digestor.digest(
                source="Long article about climate change...",
                query="What are the economic impacts of climate change?",
                source_id="doc-123",
                quality=SourceQuality.HIGH,
            )
            if result.success:
                print(result.payload.summary)
        """
        start_time = time.perf_counter()
        warnings: list[str] = []

        # Check eligibility based on policy, size, and quality
        if not self._is_eligible(source, quality):
            skip_reason = self._get_skip_reason(source, quality)
            duration_ms = self._elapsed_ms(start_time)

            # Emit metrics for skipped digest
            _metrics.counter(
                "digest_sources_processed",
                labels={"policy": self.config.policy.value, "outcome": "skipped"},
            )
            _metrics.histogram(
                "digest_duration_seconds",
                duration_ms / 1000.0,
                labels={"policy": self.config.policy.value, "outcome": "skipped"},
            )

            return DigestResult(
                payload=None,
                cache_hit=False,
                duration_ms=duration_ms,
                skipped=True,
                skip_reason=skip_reason,
            )

        try:
            # Normalize content to canonical form
            if page_boundaries:
                canonical_text, canonical_page_boundaries = self._canonicalize_pages(
                    source,
                    page_boundaries,
                )
            else:
                canonical_text = self._normalize_text(source)
                canonical_page_boundaries = None

            # Compute query hash for cache keying
            query_hash = self._compute_query_hash(query)

            # Check cache if source_id provided
            # Cache reads are allowed even when circuit breaker is open
            if source_id is not None:
                cached = self._get_cached_digest(source_id, canonical_text, query_hash)
                if cached is not None:
                    cached.duration_ms = self._elapsed_ms(start_time)

                    # Emit metrics for cache hit
                    _metrics.counter(
                        "digest_cache_hits",
                        labels={"policy": self.config.policy.value},
                    )
                    _metrics.counter(
                        "digest_sources_processed",
                        labels={"policy": self.config.policy.value, "outcome": "cache_hit"},
                    )
                    _metrics.histogram(
                        "digest_duration_seconds",
                        cached.duration_ms / 1000.0,
                        labels={"policy": self.config.policy.value, "outcome": "cache_hit"},
                    )

                    return cached

            # Check circuit breaker AFTER cache (cache reads allowed when open)
            if self._is_circuit_breaker_open():
                duration_ms = self._elapsed_ms(start_time)
                logger.debug(
                    "Digest skipped due to circuit breaker (open for %.1fs)",
                    time.time() - (self._circuit_breaker_opened_at or time.time()),
                )

                # Emit metrics for circuit breaker skip
                _metrics.counter(
                    "digest_sources_processed",
                    labels={"policy": self.config.policy.value, "outcome": "circuit_breaker"},
                )
                _metrics.histogram(
                    "digest_duration_seconds",
                    duration_ms / 1000.0,
                    labels={"policy": self.config.policy.value, "outcome": "circuit_breaker"},
                )

                return DigestResult(
                    payload=None,
                    cache_hit=False,
                    duration_ms=duration_ms,
                    skipped=True,
                    skip_reason="circuit_breaker_open",
                    warnings=["Digest skipped: circuit breaker open due to recent failures"],
                )

            # Compute source text hash for archival linkage
            source_text_hash = self._compute_source_hash(canonical_text)

            # Generate query-conditioned summary using ContentSummarizer
            # Pass query as context to focus summary on relevant aspects
            # Explicit error handling: on summarization failure, skip digest and preserve original
            try:
                summary_result = await self.summarizer.summarize_with_result(
                    canonical_text,
                    level=SummarizationLevel.KEY_POINTS,
                    context=f"Focus on aspects relevant to: {query}",
                )
            except Exception as summarization_error:
                # Summarization failed - skip digest gracefully, preserve original content
                duration_ms = self._elapsed_ms(start_time)
                logger.warning(
                    "Summarization failed, skipping digest: %s", summarization_error
                )

                # Record failure for circuit breaker tracking
                self._record_failure()

                # Emit metrics for summarization failure
                _metrics.counter(
                    "digest_sources_processed",
                    labels={"policy": self.config.policy.value, "outcome": "summarization_error"},
                )
                _metrics.histogram(
                    "digest_duration_seconds",
                    duration_ms / 1000.0,
                    labels={"policy": self.config.policy.value, "outcome": "summarization_error"},
                )

                # Return skipped result with warning - original content preserved by caller
                return DigestResult(
                    payload=None,
                    cache_hit=False,
                    duration_ms=duration_ms,
                    skipped=True,
                    skip_reason="summarization_failed",
                    warnings=[f"Summarization failed: {summarization_error}"],
                )

            # Extract summary and key points from result
            summary = summary_result.content[:self.config.max_summary_length]
            raw_key_points = summary_result.key_points[:self.config.max_key_points]
            # Enforce per-item max length (500 chars) to avoid payload validation failures
            key_points = [
                kp[:500]
                for kp in raw_key_points
                if kp and kp.strip()
            ]

            # Collect warnings from summarization
            warnings.extend(summary_result.warnings)

            # Extract evidence snippets with scoring and locators (if enabled)
            if self.config.include_evidence:
                evidence_snippets = self._build_evidence_snippets(
                    canonical_text=canonical_text,
                    query=query,
                    page_boundaries=canonical_page_boundaries,
                )
            else:
                evidence_snippets = []

            # Calculate metrics
            original_chars = len(canonical_text)
            evidence_chars = sum(len(e.text) for e in evidence_snippets)
            digest_chars = len(summary) + sum(len(kp) for kp in key_points) + evidence_chars
            compression_ratio = digest_chars / original_chars if original_chars > 0 else 1.0

            # Create DigestPayload
            payload = DigestPayload(
                query_hash=query_hash,
                summary=summary,
                key_points=key_points,
                evidence_snippets=evidence_snippets,
                original_chars=original_chars,
                digest_chars=digest_chars,
                compression_ratio=min(compression_ratio, 1.0),
                source_text_hash=source_text_hash,
            )

            logger.debug(
                f"Digest generated: {original_chars} chars -> {digest_chars} chars "
                f"({compression_ratio:.1%} compression), {len(key_points)} key points"
            )

            duration_ms = self._elapsed_ms(start_time)
            result = DigestResult(
                payload=payload,
                cache_hit=False,
                duration_ms=duration_ms,
                warnings=warnings,
            )

            # Emit metrics for successful digest
            _metrics.counter(
                "digest_sources_processed",
                labels={"policy": self.config.policy.value, "outcome": "success"},
            )
            _metrics.histogram(
                "digest_duration_seconds",
                duration_ms / 1000.0,
                labels={"policy": self.config.policy.value, "outcome": "success"},
            )
            _metrics.histogram(
                "digest_compression_ratio",
                min(compression_ratio, 1.0),
                labels={"policy": self.config.policy.value},
            )
            _metrics.histogram(
                "digest_evidence_snippets",
                len(evidence_snippets),
                labels={"policy": self.config.policy.value},
            )

            # Cache successful result if source_id provided
            if source_id is not None:
                self._cache_digest(source_id, canonical_text, query_hash, result)

            # Record success for circuit breaker tracking
            self._record_success()

            return result

        except Exception as e:
            duration_ms = self._elapsed_ms(start_time)
            logger.error(f"Digest generation failed: {e}")

            # Record failure for circuit breaker tracking
            self._record_failure()

            # Emit metrics for failed digest
            _metrics.counter(
                "digest_sources_processed",
                labels={"policy": self.config.policy.value, "outcome": "error"},
            )
            _metrics.histogram(
                "digest_duration_seconds",
                duration_ms / 1000.0,
                labels={"policy": self.config.policy.value, "outcome": "error"},
            )

            return DigestResult(
                payload=None,
                cache_hit=False,
                duration_ms=duration_ms,
                warnings=[f"Digest generation failed: {e}"],
            )

    def _is_eligible(
        self,
        content: str,
        quality: Optional[SourceQuality] = None,
    ) -> bool:
        """Check if content is eligible for digestion based on policy.

        Applies the configured digest policy to determine eligibility:
        - OFF: Always returns False (no digestion)
        - ALWAYS: Returns True if content is non-empty
        - AUTO: Checks size threshold and quality filter

        For AUTO policy, quality must be HIGH or MEDIUM (or above the
        configured quality_threshold). Sources with LOW or UNKNOWN quality
        are not digested in AUTO mode.

        Args:
            content: Content to check.
            quality: Optional source quality level. If not provided for AUTO
                policy, defaults to checking only size threshold.

        Returns:
            True if content is eligible for digestion.

        Examples:
            # OFF policy - never eligible
            >>> config = DigestConfig(policy=DigestPolicy.OFF)
            >>> digestor._is_eligible("content", SourceQuality.HIGH)
            False

            # ALWAYS policy - eligible if non-empty
            >>> config = DigestConfig(policy=DigestPolicy.ALWAYS)
            >>> digestor._is_eligible("content", SourceQuality.LOW)
            True

            # AUTO policy - checks size and quality
            >>> config = DigestConfig(policy=DigestPolicy.AUTO, min_content_length=100)
            >>> digestor._is_eligible("A" * 200, SourceQuality.HIGH)
            True
            >>> digestor._is_eligible("A" * 200, SourceQuality.LOW)
            False
        """
        # OFF policy: never digest
        if self.config.policy == DigestPolicy.OFF:
            return False

        # ALWAYS policy: digest any non-empty content
        if self.config.policy == DigestPolicy.ALWAYS:
            return bool(content and content.strip())

        # AUTO policy: check size and quality thresholds
        # Check size threshold
        if len(content) < self.config.min_content_length:
            return False

        # Check quality threshold - required for AUTO policy
        # Missing quality (None) is treated as UNKNOWN and rejected by default
        # Quality hierarchy: HIGH > MEDIUM > LOW > UNKNOWN
        quality_order = {
            SourceQuality.HIGH: 3,
            SourceQuality.MEDIUM: 2,
            SourceQuality.LOW: 1,
            SourceQuality.UNKNOWN: 0,
        }
        threshold_level = quality_order.get(self.config.quality_threshold, 2)

        # Treat None as UNKNOWN (level 0), which fails default MEDIUM threshold
        source_level = quality_order.get(quality, 0) if quality is not None else 0

        if source_level < threshold_level:
            return False

        return True

    def _get_skip_reason(
        self,
        content: str,
        quality: Optional[SourceQuality] = None,
    ) -> str:
        """Generate a human-readable skip reason for ineligible content.

        Args:
            content: Content that was checked.
            quality: Optional source quality level.

        Returns:
            Descriptive reason why content was skipped.
        """
        if self.config.policy == DigestPolicy.OFF:
            return "Digest policy is OFF"

        if self.config.policy == DigestPolicy.ALWAYS:
            return "Content is empty"

        # AUTO policy - determine specific reason
        if len(content) < self.config.min_content_length:
            return (
                f"Content length ({len(content)}) below minimum "
                f"({self.config.min_content_length})"
            )

        # Check quality - None is treated as missing/unknown
        quality_order = {
            SourceQuality.HIGH: 3,
            SourceQuality.MEDIUM: 2,
            SourceQuality.LOW: 1,
            SourceQuality.UNKNOWN: 0,
        }
        threshold_level = quality_order.get(self.config.quality_threshold, 2)
        source_level = quality_order.get(quality, 0) if quality is not None else 0

        if source_level < threshold_level:
            if quality is None:
                return (
                    f"Source quality not provided (required for AUTO policy, "
                    f"minimum: {self.config.quality_threshold.value})"
                )
            return (
                f"Source quality ({quality.value}) below threshold "
                f"({self.config.quality_threshold.value})"
            )

        return "Content not eligible for digest"

    def _normalize_text(self, text: str) -> str:
        """Normalize text to canonical form.

        Applies a deterministic normalization pipeline to ensure consistent
        hashing and text processing. The pipeline is designed to be
        idempotent - applying it multiple times produces the same result.

        Normalization steps (in order):
        1. HTML entity decoding (&amp; -> &, &lt; -> <, etc.)
        2. HTML tag stripping (removes <tag> and </tag>)
        3. Unicode normalization to NFC form
        4. Whitespace collapse (multiple spaces/newlines -> single space)

        Args:
            text: Raw text to normalize.

        Returns:
            Normalized canonical text suitable for hashing and evidence extraction.

        Examples:
            >>> digestor._normalize_text("Hello&nbsp;World")
            'Hello World'
            >>> digestor._normalize_text("<p>Hello</p> <b>World</b>")
            'Hello World'
            >>> digestor._normalize_text("Hello\\n\\n\\nWorld")
            'Hello World'
        """
        return self._canonicalize_text(text)

    def _canonicalize_pages(
        self,
        text: str,
        page_boundaries: list[tuple[int, int, int]],
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Canonicalize text while preserving PDF page boundary mapping.

        Args:
            text: Raw source text.
            page_boundaries: List of (page_num, start, end) offsets into raw text.

        Returns:
            Tuple of (canonical_text, canonical_page_boundaries).
        """
        canonical_pages: list[str] = []
        canonical_bounds: list[tuple[int, int, int]] = []
        cursor = 0

        for page_num, start, end in page_boundaries:
            page_text = text[start:end]
            page_canonical = self._canonicalize_text(page_text)

            if canonical_pages:
                cursor += 2  # Account for "\n\n" separator between pages

            page_start = cursor
            page_end = page_start + len(page_canonical)
            canonical_bounds.append((page_num, page_start, page_end))
            canonical_pages.append(page_canonical)
            cursor = page_end

        canonical_text = "\n\n".join(canonical_pages)
        return canonical_text, canonical_bounds

    def _canonicalize_text(self, text: str) -> str:
        """Apply canonical text normalization pipeline.

        This is the core normalization implementation. The method is separate
        from _normalize_text to allow direct access for testing while
        maintaining the existing public interface.

        Normalization pipeline:
        1. Decode HTML entities (&amp; -> &, &lt; -> <, &nbsp; -> space, etc.)
        2. Strip HTML tags (both opening and closing)
        3. Normalize Unicode to NFC form (composed characters)
        4. Collapse whitespace (multiple spaces/newlines/tabs -> single space)
        5. Strip leading/trailing whitespace

        Args:
            text: Raw text to normalize.

        Returns:
            Canonical text form.
        """
        if not text:
            return ""

        # Step 1: Decode HTML entities
        # Handles &amp; &lt; &gt; &quot; &nbsp; and numeric entities like &#39;
        result = html.unescape(text)

        # Step 2: Strip HTML tags
        # Simple regex that handles <tag>, </tag>, <tag attr="value">, etc.
        result = re.sub(r"<[^>]+>", " ", result)

        # Step 3: Unicode normalization to NFC
        # NFC is the canonical form for text comparison
        # Composes characters (e.g., 'é' as single codepoint vs e + combining accent)
        result = unicodedata.normalize("NFC", result)

        # Step 4: Collapse whitespace
        # Replace all whitespace sequences (spaces, tabs, newlines) with single space
        result = re.sub(r"\s+", " ", result)

        # Step 5: Strip leading/trailing whitespace
        result = result.strip()

        return result

    def _compute_query_hash(self, query: str) -> str:
        """Compute 8-character hex hash of the query.

        Args:
            query: Research query string.

        Returns:
            8-character lowercase hex hash.
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]

    def _compute_source_hash(self, canonical_text: str) -> str:
        """Compute SHA256 hash of canonical text with prefix.

        Args:
            canonical_text: Normalized source text.

        Returns:
            Hash string in format "sha256:{64-char-hex}".
        """
        hash_hex = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        return f"sha256:{hash_hex}"

    def _elapsed_ms(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds.

        Args:
            start_time: Start time from time.perf_counter().

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.perf_counter() - start_time) * 1000

    def _chunk_text(
        self,
        text: str,
        *,
        target_size: int = 400,
        max_size: int = 500,
        min_size: int = 50,
    ) -> list[str]:
        """Chunk text into segments for evidence extraction.

        Splits text into chunks using boundary-aware logic that respects
        natural text boundaries when possible. Chunks target a specific
        size but will extend to reach a clean boundary up to max_size.
        Small trailing chunks below min_size are merged with the previous.

        Boundary detection priority (highest to lowest):
        1. Paragraph boundaries (double newline or blank line)
        2. Sentence boundaries (. ! ? followed by space or end)
        3. Clause boundaries (, ; : followed by space)
        4. Word boundaries (space)
        5. Hard cut (last resort at max_size)

        Args:
            text: Text to chunk.
            target_size: Target chunk size in characters. Default 400.
            max_size: Maximum chunk size before hard cut. Default 500.
            min_size: Minimum chunk size; smaller chunks merge. Default 50.

        Returns:
            List of text chunks. May be empty if input is empty/whitespace.

        Examples:
            >>> digestor._chunk_text("Short text")
            ['Short text']
            >>> chunks = digestor._chunk_text("First paragraph.\\n\\nSecond paragraph.")
            >>> len(chunks) >= 1
            True
        """
        if not text or not text.strip():
            return []

        # Ensure text is normalized (no leading/trailing whitespace)
        text = text.strip()

        # If text fits within target, return as single chunk
        if len(text) <= target_size:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            # If remaining text fits in target, add it and stop
            if len(remaining) <= target_size:
                chunks.append(remaining)
                break

            # Find the best boundary within max_size
            chunk_end = self._find_chunk_boundary(
                remaining,
                target_size=target_size,
                max_size=max_size,
            )

            # Extract chunk and strip
            chunk = remaining[:chunk_end].strip()
            remaining = remaining[chunk_end:].strip()

            if chunk:
                chunks.append(chunk)

        # Merge small final chunk with previous if below min_size
        if len(chunks) >= 2 and len(chunks[-1]) < min_size:
            merged = chunks[-2] + " " + chunks[-1]
            # Only merge if result doesn't exceed max_size
            if len(merged) <= max_size:
                chunks[-2] = merged
                chunks.pop()

        return chunks

    def _find_chunk_boundary(
        self,
        text: str,
        *,
        target_size: int,
        max_size: int,
    ) -> int:
        """Find the best boundary position for chunking.

        Searches for natural text boundaries starting from target_size
        up to max_size. Returns the position immediately after the
        boundary marker (so the marker is included in the chunk).

        Boundary priority:
        1. Paragraph (\\n\\n) - look backward from target first
        2. Sentence (. ! ?) - followed by space or at end
        3. Clause (, ; :) - followed by space
        4. Word (space)
        5. Hard cut at max_size

        Args:
            text: Text to find boundary in.
            target_size: Start searching from this position.
            max_size: Maximum position (hard cut fallback).

        Returns:
            Position to cut at (exclusive).
        """
        # Clamp max_size to actual text length
        effective_max = min(max_size, len(text))
        effective_target = min(target_size, len(text))

        # Priority 1: Paragraph boundary (double newline)
        # Look backward from target first, then forward to max
        para_pos = self._find_boundary_bidirectional(
            text,
            patterns=["\n\n", "\r\n\r\n"],
            target=effective_target,
            max_pos=effective_max,
        )
        if para_pos > 0:
            return para_pos

        # Priority 2: Sentence boundary (. ! ? followed by space or at end)
        sent_pos = self._find_sentence_boundary(
            text,
            target=effective_target,
            max_pos=effective_max,
        )
        if sent_pos > 0:
            return sent_pos

        # Priority 3: Clause boundary (; : , followed by space)
        clause_pos = self._find_boundary_bidirectional(
            text,
            patterns=["; ", ": ", ", "],
            target=effective_target,
            max_pos=effective_max,
            include_pattern=True,
        )
        if clause_pos > 0:
            return clause_pos

        # Priority 4: Word boundary (space)
        word_pos = self._find_boundary_bidirectional(
            text,
            patterns=[" "],
            target=effective_target,
            max_pos=effective_max,
            include_pattern=False,
        )
        if word_pos > 0:
            return word_pos

        # Priority 5: Hard cut at max_size
        return effective_max

    def _find_boundary_bidirectional(
        self,
        text: str,
        patterns: list[str],
        target: int,
        max_pos: int,
        include_pattern: bool = True,
    ) -> int:
        """Find boundary pattern, searching backward from target then forward.

        Args:
            text: Text to search.
            patterns: Pattern strings to look for.
            target: Start position for search.
            max_pos: Maximum position to search forward.
            include_pattern: If True, include pattern length in result.

        Returns:
            Position after boundary, or 0 if not found.
        """
        best_backward = 0
        best_forward = 0

        for pattern in patterns:
            # Search backward from target
            backward = text.rfind(pattern, 0, target)
            if backward > best_backward:
                if include_pattern:
                    best_backward = backward + len(pattern)
                else:
                    best_backward = backward

            # Search forward from target to max_pos
            forward = text.find(pattern, target, max_pos)
            if forward > 0 and (best_forward == 0 or forward < best_forward):
                if include_pattern:
                    best_forward = forward + len(pattern)
                else:
                    best_forward = forward

        # Prefer backward result if found and reasonably close to target
        # (within 100 chars), otherwise take forward if available
        if best_backward > 0 and target - best_backward <= 100:
            return best_backward
        if best_forward > 0:
            return best_forward
        if best_backward > 0:
            return best_backward

        return 0

    def _find_sentence_boundary(
        self,
        text: str,
        target: int,
        max_pos: int,
    ) -> int:
        """Find sentence boundary (. ! ? followed by space or at end).

        Handles edge cases like abbreviations by requiring space after
        punctuation (except at text end).

        Args:
            text: Text to search.
            target: Start position for search.
            max_pos: Maximum position.

        Returns:
            Position after sentence end, or 0 if not found.
        """
        sentence_markers = ".!?"

        # Search backward from target
        best_backward = 0
        for i in range(target - 1, -1, -1):
            if text[i] in sentence_markers:
                # Check if followed by space or at end
                if i + 1 >= len(text) or text[i + 1] in " \n\t":
                    best_backward = i + 1
                    break

        # Search forward from target to max_pos
        best_forward = 0
        for i in range(target, min(max_pos, len(text))):
            if text[i] in sentence_markers:
                # Check if followed by space or at end
                if i + 1 >= len(text) or text[i + 1] in " \n\t":
                    best_forward = i + 1
                    break

        # Prefer backward if reasonably close (within 100 chars)
        if best_backward > 0 and target - best_backward <= 100:
            return best_backward
        if best_forward > 0:
            return best_forward
        if best_backward > 0:
            return best_backward

        return 0

    def _extract_evidence(
        self,
        text: str,
        query: str,
        *,
        max_snippets: Optional[int] = None,
    ) -> list[tuple[str, int, float]]:
        """Extract evidence snippets from text based on query relevance.

        Chunks the text and scores each chunk based on query term matching.
        Returns the top-scoring chunks as evidence snippets with their
        original position and relevance score.

        Scoring formula:
        - For each query term found in chunk (case-insensitive):
          score += 1 / (1 + log(term_frequency_in_corpus))
        - This gives higher weight to rarer terms

        Tie-breakers (applied in order):
        1. Higher score wins
        2. Earlier position wins (lower index)
        3. Longer chunk wins (more context)

        Empty/short query fallback:
        - If query is empty or < 3 chars, uses positional scoring
        - Early chunks get higher scores (1.0 - position/total)

        Args:
            text: Source text to extract evidence from.
            query: Research query to match against.
            max_snippets: Maximum number of snippets to return.
                Defaults to config.max_evidence_snippets.

        Returns:
            List of tuples (snippet_text, position_index, score).
            Sorted by score descending, then position ascending.

        Examples:
            >>> evidence = digestor._extract_evidence(
            ...     "Climate change affects coastal cities. Rising seas threaten infrastructure.",
            ...     "climate coastal impact"
            ... )
            >>> len(evidence) <= digestor.config.max_evidence_snippets
            True
        """
        if max_snippets is None:
            max_snippets = self.config.max_evidence_snippets

        # Chunk the text using configured sizing constraints
        target_size = min(self.config.chunk_size, self.config.max_snippet_length)
        chunks = self._chunk_text(
            text,
            target_size=target_size,
            max_size=self.config.max_snippet_length,
            min_size=min(50, self.config.max_snippet_length),
        )
        if not chunks:
            return []

        # Handle empty/short query with positional fallback
        if not query or len(query.strip()) < 3:
            return self._score_by_position(chunks, max_snippets)

        # Extract and normalize query terms
        query_terms = self._extract_terms(query)
        if not query_terms:
            return self._score_by_position(chunks, max_snippets)

        # Calculate corpus term frequencies for IDF-like weighting
        corpus_text = text.lower()
        term_frequencies = {}
        for term in query_terms:
            term_frequencies[term] = corpus_text.count(term.lower())

        # Score each chunk
        scored_chunks: list[tuple[str, int, float, int]] = []
        for idx, chunk in enumerate(chunks):
            score = self._score_chunk(chunk, query_terms, term_frequencies)
            # Store: (chunk, position, score, length) for tie-breaking
            scored_chunks.append((chunk, idx, score, len(chunk)))

        # Sort by: score DESC, position ASC, length DESC
        scored_chunks.sort(key=lambda x: (-x[2], x[1], -x[3]))

        # Return top N as (text, position, score)
        return [(chunk, pos, score) for chunk, pos, score, _ in scored_chunks[:max_snippets]]

    def _extract_terms(self, query: str) -> list[str]:
        """Extract normalized terms from query for matching.

        Splits query on whitespace and punctuation, lowercases,
        and filters out stopwords and very short terms.

        Args:
            query: Query string to extract terms from.

        Returns:
            List of normalized query terms.
        """
        # Common English stopwords to filter out
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must",
            "that", "which", "who", "whom", "this", "these", "those",
            "it", "its", "as", "if", "when", "where", "how", "what", "why",
        }

        # Split on non-alphanumeric characters
        raw_terms = re.split(r"[^a-zA-Z0-9]+", query.lower())

        # Filter: remove stopwords and terms < 2 chars
        terms = [
            term for term in raw_terms
            if term and len(term) >= 2 and term not in stopwords
        ]

        return terms

    def _score_chunk(
        self,
        chunk: str,
        query_terms: list[str],
        term_frequencies: dict[str, int],
    ) -> float:
        """Score a chunk based on query term matches.

        Uses a term matching formula where each matched term contributes
        to the score with IDF-inspired weighting: rarer terms in the
        corpus contribute more to relevance.

        Formula: score += 1 / (1 + log(corpus_frequency + 1))

        Args:
            chunk: Text chunk to score.
            query_terms: Normalized query terms.
            term_frequencies: Term -> corpus count mapping.

        Returns:
            Relevance score (higher = more relevant).
        """
        import math

        chunk_lower = chunk.lower()
        score = 0.0

        for term in query_terms:
            if term in chunk_lower:
                corpus_freq = term_frequencies.get(term, 0)

                # IDF-inspired weighting: rarer terms score higher
                term_weight = 1.0 / (1.0 + math.log(corpus_freq + 1))
                score += term_weight

        return score

    def _score_by_position(
        self,
        chunks: list[str],
        max_snippets: int,
    ) -> list[tuple[str, int, float]]:
        """Score chunks by position (fallback for empty/short queries).

        Earlier chunks get higher scores, assuming important content
        tends to appear early in documents.

        Args:
            chunks: List of text chunks.
            max_snippets: Maximum snippets to return.

        Returns:
            List of (text, position, score) sorted by position.
        """
        total = len(chunks)
        results: list[tuple[str, int, float]] = []

        for idx, chunk in enumerate(chunks):
            # Score decreases linearly with position
            # First chunk = 1.0, last chunk = 1/total
            score = 1.0 - (idx / total) if total > 1 else 1.0
            results.append((chunk, idx, score))

        # Already sorted by position (ascending), take top N
        return results[:max_snippets]

    def _build_evidence_snippets(
        self,
        canonical_text: str,
        query: str,
        *,
        page_boundaries: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[EvidenceSnippet]:
        """Build evidence snippets with scoring and locators.

        Orchestrates the evidence extraction pipeline:
        1. Extract and score evidence chunks from canonical text
        2. Generate locators for each chunk
        3. Construct EvidenceSnippet objects with all metadata

        Args:
            canonical_text: Normalized source text.
            query: Research query for relevance scoring.
            page_boundaries: Optional PDF page boundaries for locators.

        Returns:
            List of EvidenceSnippet objects, limited by config.max_evidence_snippets.
        """
        if not self.config.include_evidence:
            return []

        # Extract evidence with relevance scoring
        evidence_tuples = self._extract_evidence(
            canonical_text,
            query,
            max_snippets=self.config.max_evidence_snippets,
        )

        if not evidence_tuples:
            return []

        # Generate locators in original text order to keep search positions valid,
        # then map back to relevance order.
        indexed_tuples = list(enumerate(evidence_tuples))
        indexed_tuples.sort(key=lambda item: item[1][1])  # sort by position index

        ordered_texts = [text for _, (text, _, _) in indexed_tuples]
        ordered_locators = self._generate_locators_batch(
            canonical_text,
            ordered_texts,
            page_boundaries=page_boundaries,
        )

        locators_by_index: list[tuple[str, int, int]] = [("char:0-0", 0, 0)] * len(
            evidence_tuples
        )
        for ordered_idx, (original_idx, _) in enumerate(indexed_tuples):
            locators_by_index[original_idx] = ordered_locators[ordered_idx]

        # Build EvidenceSnippet objects
        # Note: No truncation applied here - chunks already respect max_size (500)
        # from _chunk_text(). Display truncation is applied at render time per spec.
        snippets: list[EvidenceSnippet] = []
        for i, (text, _, score) in enumerate(evidence_tuples):
            locator_str, _, _ = locators_by_index[i]

            # Normalize score to 0.0-1.0 range
            # (scores from _extract_evidence may exceed 1.0)
            normalized_score = min(1.0, max(0.0, score))

            snippets.append(
                EvidenceSnippet(
                    text=text,
                    locator=locator_str,
                    relevance_score=normalized_score,
                )
            )

        return snippets

    def _generate_locator(
        self,
        canonical_text: str,
        snippet_text: str,
        search_start: int = 0,
        *,
        page_number: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Generate a locator string for a text snippet.

        Creates a locator that uniquely identifies the snippet's position
        within the canonical text. The locator format allows direct
        retrieval: canonical_text[start:end] == snippet_text.

        Locator formats:
        - Text: "char:{start}-{end}" (e.g., "char:100-250")
        - PDF: "page:{n}:char:{start}-{end}" (e.g., "page:3:char:100-250")

        Offset conventions:
        - start: 0-based index of first character
        - end: exclusive (Python slice convention)
        - Page numbers are 1-based (human-readable)

        Args:
            canonical_text: The normalized source text to search.
            snippet_text: The exact snippet text to locate.
            search_start: Position to start searching from (for efficiency
                when locating multiple snippets in order).
            page_number: Optional 1-based page number for PDF sources.
                If provided, generates page-prefixed locator.

        Returns:
            Tuple of (locator_string, start_offset, end_offset).
            If snippet not found, returns ("char:0-0", 0, 0).

        Examples:
            >>> text = "The quick brown fox jumps over the lazy dog."
            >>> locator, start, end = digestor._generate_locator(text, "brown fox")
            >>> locator
            'char:10-19'
            >>> text[start:end]
            'brown fox'

            >>> locator, _, _ = digestor._generate_locator(text, "fox", page_number=2)
            >>> locator
            'page:2:char:16-19'
        """
        # Find the snippet in the canonical text
        start = canonical_text.find(snippet_text, search_start)

        if start == -1:
            # Snippet not found - return null locator
            snippet_hash = hashlib.sha256(snippet_text.encode("utf-8")).hexdigest()[:8]
            logger.warning(
                "Snippet not found in canonical text (len=%d, hash=%s)",
                len(snippet_text),
                snippet_hash,
            )
            return ("char:0-0", 0, 0)

        end = start + len(snippet_text)

        # Build locator string
        if page_number is not None:
            locator = f"page:{page_number}:char:{start}-{end}"
        else:
            locator = f"char:{start}-{end}"

        return (locator, start, end)

    def _generate_locators_batch(
        self,
        canonical_text: str,
        snippets: list[str],
        *,
        page_boundaries: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[tuple[str, int, int]]:
        """Generate locators for multiple snippets efficiently.

        Processes snippets in order, using the previous end position as
        the search start for better performance on large texts.

        For PDF sources with page boundaries, automatically determines
        which page each snippet belongs to and includes it in the locator.

        Args:
            canonical_text: The normalized source text.
            snippets: List of snippet texts to locate.
            page_boundaries: Optional list of (page_num, start_char, end_char)
                tuples defining page boundaries in the canonical text.
                Page numbers should be 1-based.

        Returns:
            List of (locator, start, end) tuples, one per snippet.
            Order matches input snippets list.

        Examples:
            >>> locators = digestor._generate_locators_batch(
            ...     "First chunk. Second chunk. Third chunk.",
            ...     ["First chunk", "Second chunk", "Third chunk"]
            ... )
            >>> len(locators) == 3
            True
        """
        results: list[tuple[str, int, int]] = []
        search_pos = 0

        for snippet in snippets:
            # Determine page number if boundaries provided
            page_num = None
            if page_boundaries:
                # Find which page contains the expected position
                for pnum, pstart, pend in page_boundaries:
                    # First try to find snippet starting from search_pos
                    test_start = canonical_text.find(snippet, search_pos)
                    if test_start >= pstart and test_start < pend:
                        page_num = pnum
                        break

            locator, start, end = self._generate_locator(
                canonical_text,
                snippet,
                search_start=search_pos,
                page_number=page_num,
            )

            results.append((locator, start, end))

            # Update search position for next snippet (if found)
            if end > 0:
                search_pos = end

        return results

    def generate_cache_key(
        self,
        source_id: str,
        content_hash: str,
        query_hash: str,
        config_hash: str,
        *,
        summarizer_hash: Optional[str] = None,
        impl_version: str = DIGEST_IMPL_VERSION,
    ) -> str:
        """Generate a cache key for digest results.

        Creates a unique cache key that incorporates all factors affecting
        digest output: implementation version, source identity, content,
        query, configuration, and summarizer configuration. Any change to
        these factors produces a different cache key, ensuring cache
        invalidation on changes.

        Key format:
            digest:{impl_version}:{source_id}:{content_hash[:16]}:{query_hash[:8]}:{config_hash[:8]}:{summarizer_hash[:8]}

        Hash truncations balance uniqueness with key length:
        - content_hash[:16]: 16 hex chars (64 bits) - primary content identity
        - query_hash[:8]: 8 hex chars (32 bits) - query conditioning
        - config_hash[:8]: 8 hex chars (32 bits) - configuration variant
        - summarizer_hash[:8]: 8 hex chars (32 bits) - summarizer config variant

        Args:
            source_id: Unique identifier for the source document.
            content_hash: Full SHA256 hash of canonical content (sha256:... format).
            query_hash: 8-char hex hash of the research query.
            config_hash: Hash of digest configuration.
            summarizer_hash: Optional hash of summarizer configuration. If not
                provided, computed from the current summarizer settings.
            impl_version: Digest implementation version. Default "1.0".

        Returns:
            Cache key string in specified format.

        Examples:
            >>> key = digestor.generate_cache_key(
            ...     source_id="doc-123",
            ...     content_hash="sha256:abcd1234...",
            ...     query_hash="ef567890",
            ...     config_hash="12345678abcdef00",
            ... )
            >>> key
            'digest:1.0:doc-123:abcd1234567890ab:ef567890:12345678:deadbeef'
        """
        # Extract hex portion from content_hash if it has sha256: prefix
        if content_hash.startswith("sha256:"):
            content_hex = content_hash[7:]  # Remove "sha256:" prefix
        else:
            content_hex = content_hash

        # Truncate hashes per spec
        content_truncated = content_hex[:16]
        query_truncated = query_hash[:8]
        config_truncated = config_hash[:8]
        if summarizer_hash is None:
            summarizer_hash = self._compute_summarizer_hash()
        summarizer_truncated = summarizer_hash[:8]

        return (
            f"digest:{impl_version}:{source_id}:"
            f"{content_truncated}:{query_truncated}:{config_truncated}:{summarizer_truncated}"
        )

    def _get_cached_digest(
        self,
        source_id: str,
        canonical_text: str,
        query_hash: str,
    ) -> Optional[DigestResult]:
        """Check cache for existing digest result.

        Args:
            source_id: Source document identifier.
            canonical_text: Normalized source text.
            query_hash: Hash of the research query.

        Returns:
            Cached DigestResult with cache_hit=True, or None if not cached.
        """
        content_hash = self._compute_source_hash(canonical_text)
        config_hash = self.config.compute_config_hash()
        cache_key = self.generate_cache_key(source_id, content_hash, query_hash, config_hash)

        cached = self._cache.get(cache_key)
        if cached is not None:
            # Return copy with cache_hit flag set
            return DigestResult(
                payload=cached.payload,
                cache_hit=True,
                duration_ms=cached.duration_ms,
                skipped=cached.skipped,
                skip_reason=cached.skip_reason,
                warnings=cached.warnings,
            )
        return None

    def _cache_digest(
        self,
        source_id: str,
        canonical_text: str,
        query_hash: str,
        result: DigestResult,
    ) -> None:
        """Store digest result in cache.

        Args:
            source_id: Source document identifier.
            canonical_text: Normalized source text.
            query_hash: Hash of the research query.
            result: DigestResult to cache.
        """
        content_hash = self._compute_source_hash(canonical_text)
        config_hash = self.config.compute_config_hash()
        cache_key = self.generate_cache_key(source_id, content_hash, query_hash, config_hash)
        self._cache.set(cache_key, result)

    def _compute_summarizer_hash(self) -> str:
        """Compute a hash for the summarizer configuration.

        Includes summarizer class identity, provider chain, and key
        configuration fields to ensure cache invalidation when the
        summarizer behavior changes.
        """
        summarizer = self.summarizer
        summarizer_id = (
            f"{summarizer.__class__.__module__}."
            f"{summarizer.__class__.__qualname__}"
        )
        provider_func = getattr(summarizer, "_provider_func", None)
        provider_func_name = None
        if provider_func is not None and provider_func.__class__.__module__ != "unittest.mock":
            provider_func_name = getattr(
                provider_func,
                "__qualname__",
                getattr(provider_func, "__name__", "custom_provider"),
            )

        config = getattr(summarizer, "config", None)
        if config is not None and config.__class__.__module__ == "unittest.mock":
            config = None
        provider_chain: list[str] = []
        if config is not None and hasattr(config, "get_provider_chain"):
            try:
                chain = config.get_provider_chain()
            except Exception:
                chain = []
            if isinstance(chain, (list, tuple)):
                provider_chain = list(chain)
            else:
                provider_chain = []

        config_tuple = (
            summarizer_id,
            tuple(provider_chain),
            getattr(config, "max_retries", None),
            getattr(config, "retry_delay", None),
            getattr(config, "timeout", None),
            getattr(config, "chunk_size", None),
            getattr(config, "chunk_overlap", None),
            getattr(config, "target_budget", None),
            getattr(config, "cache_enabled", None),
            provider_func_name,
        )
        return hashlib.sha256(str(config_tuple).encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Serialization Functions
# =============================================================================


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
