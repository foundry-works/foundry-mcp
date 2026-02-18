"""Content summarizer with provider chain, retry logic, and caching.

Provides the main ContentSummarizer class for summarizing content using
LLM providers with automatic fallback, chunking for large content,
and budget enforcement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from foundry_mcp.core.errors.research import (
    ProviderExhaustedError,
    SummarizationError,
)

from .cache import SummaryCache
from .constants import CHARS_PER_TOKEN, CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, MAX_RETRIES, RETRY_DELAY
from .models import SummarizationConfig, SummarizationFunc, SummarizationLevel, SummarizationResult

logger = logging.getLogger(__name__)


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
