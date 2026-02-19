"""Core DocumentDigestor class.

Orchestrates document digest generation by combining text processing,
evidence extraction, and circuit breaker mixins.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.research.models.sources import SourceQuality
from foundry_mcp.core.research.pdf_extractor import PDFExtractor
from foundry_mcp.core.research.summarization import (
    ContentSummarizer,
    SummarizationLevel,
)

from .cache import DigestCache
from .circuit_breaker import CircuitBreakerMixin
from .config import DigestConfig, DigestPolicy
from .evidence import EvidenceExtractionMixin
from .results import DigestResult
from .text_processing import TextProcessingMixin

# Digest implementation version. Bump when algorithm changes to invalidate caches.
DIGEST_IMPL_VERSION = "1.0"

# Initialize metrics collector
_metrics = get_metrics()

logger = logging.getLogger(__name__)


class DocumentDigestor(
    TextProcessingMixin,
    EvidenceExtractionMixin,
    CircuitBreakerMixin,
):
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
                logger.warning("Summarization failed, skipping digest: %s", summarization_error)

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
            summary = summary_result.content[: self.config.max_summary_length]
            raw_key_points = summary_result.key_points[: self.config.max_key_points]
            # Enforce per-item max length (500 chars) to avoid payload validation failures
            key_points = [kp[:500] for kp in raw_key_points if kp and kp.strip()]

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

            # Import here to avoid circular imports at module level
            from foundry_mcp.core.research.models.digest import DigestPayload

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
            return f"Content length ({len(content)}) below minimum ({self.config.min_content_length})"

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
            return f"Source quality ({quality.value}) below threshold ({self.config.quality_threshold.value})"

        return "Content not eligible for digest"

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
        summarizer_id = f"{summarizer.__class__.__module__}.{summarizer.__class__.__qualname__}"
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
