"""Context budget manager for token budget allocation.

Orchestrates priority-based token budget allocation across content items
using configurable allocation strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from foundry_mcp.core.research.models.sources import ResearchSource
from foundry_mcp.core.research.token_management import estimate_tokens

from .constants import MAX_TOKEN_CACHE_ENTRIES
from .models import AllocatedItem, AllocationResult, AllocationStrategy

logger = logging.getLogger(__name__)


class ContextBudgetManager:
    """Orchestrates priority-based token budget allocation.

    Manages the distribution of a token budget across multiple content
    items based on priority and allocation strategy. Tracks which items
    fit at full fidelity, which need compression, and which must be dropped.

    The manager does not perform actual summarization - it determines
    allocation targets. Use ContentSummarizer to compress items that
    have needs_summarization=True in the result.

    Attributes:
        token_estimator: Function to estimate tokens for content
        provider: Provider hint for token estimation accuracy

    Example:
        manager = ContextBudgetManager(provider="claude")

        # Prepare items (any objects implementing ContentItem protocol)
        items = [
            {"id": "src-1", "content": "...", "priority": 1},
            {"id": "src-2", "content": "...", "priority": 2},
        ]

        # Allocate budget
        result = manager.allocate_budget(
            items=items,
            budget=50_000,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Process results
        for item in result.items:
            if item.needs_summarization:
                # Summarize to fit allocated_tokens
                summarized = await summarizer.summarize(
                    item.content,
                    target_tokens=item.allocated_tokens,
                )
    """

    def __init__(
        self,
        *,
        token_estimator: Optional[Callable[[str], int]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the context budget manager.

        Args:
            token_estimator: Custom function to estimate token counts.
                If not provided, uses estimate_tokens from token_management.
            provider: Provider hint for more accurate token estimation
            model: Model hint for more accurate token estimation
        """
        self._token_estimator = token_estimator
        self._provider = provider
        self._model = model

    def _estimate_tokens(self, content: str) -> int:
        """Estimate tokens for content using configured estimator.

        Args:
            content: Text content to estimate

        Returns:
            Estimated token count
        """
        if self._token_estimator:
            return self._token_estimator(content)
        return estimate_tokens(
            content,
            provider=self._provider,
            model=self._model,
            warn_on_heuristic=False,  # Suppress repeated warnings in batch
        )

    def _get_item_tokens(self, item: Any) -> int:
        """Get or estimate token count for an item.

        For ResearchSource items, checks the internal token cache first.
        On cache miss, computes the token count and stores it with FIFO
        eviction when the cache exceeds MAX_TOKEN_CACHE_ENTRIES (50).

        Args:
            item: Content item (must have 'content' attribute)

        Returns:
            Token count (from cache, item.tokens if present, else estimated)
        """
        # Check for pre-computed tokens
        if hasattr(item, "tokens") and item.tokens is not None:
            return item.tokens

        # For ResearchSource (direct or attached to content item), check cached token count
        source_ref: Optional[ResearchSource] = None
        if isinstance(item, ResearchSource):
            source_ref = item
        else:
            candidate = getattr(item, "source_ref", None)
            if isinstance(candidate, ResearchSource):
                source_ref = candidate

        if source_ref is not None and self._provider and self._model:
            cached = source_ref._get_cached_token_count(self._provider, self._model)
            if cached is not None:
                logger.debug(
                    f"Token cache hit for {source_ref.id}: {cached} tokens "
                    f"(provider={self._provider}, model={self._model})"
                )
                return cached

        # Estimate from content
        content = getattr(item, "content", "")
        tokens = self._estimate_tokens(content)

        # For ResearchSource, store in cache with FIFO eviction
        if source_ref is not None and self._provider and self._model:
            self._store_token_count_with_eviction(source_ref, tokens)
            logger.debug(
                f"Token cache miss for {source_ref.id}: computed {tokens} tokens "
                f"(provider={self._provider}, model={self._model})"
            )

        return tokens

    def _store_token_count_with_eviction(self, source: ResearchSource, count: int) -> None:
        """Store token count in source cache with FIFO eviction.

        If the cache exceeds MAX_TOKEN_CACHE_ENTRIES, removes the oldest
        entry before adding the new one. Dict key insertion order is
        preserved in Python 3.7+, so we remove the first key for FIFO.

        Args:
            source: ResearchSource to update
            count: Token count to store
        """
        if not self._provider or not self._model:
            return

        # Ensure cache exists
        if "_token_cache" not in source.metadata:
            source.metadata["_token_cache"] = {"v": 1, "counts": {}}

        cache = source.metadata["_token_cache"]
        if "counts" not in cache:
            cache["counts"] = {}

        counts = cache["counts"]

        # FIFO eviction if at capacity
        while len(counts) >= MAX_TOKEN_CACHE_ENTRIES:
            # Remove oldest entry (first key in insertion order)
            oldest_key = next(iter(counts))
            del counts[oldest_key]
            logger.debug(f"Token cache eviction: removed {oldest_key}")

        # Store new count
        source._set_cached_token_count(self._provider, self._model, count)

    def _sort_by_priority(self, items: Sequence[Any]) -> list[Any]:
        """Sort items by priority (1 = highest, first).

        Args:
            items: Sequence of content items

        Returns:
            List sorted by priority ascending (highest priority first)
        """
        return sorted(items, key=lambda x: getattr(x, "priority", 999))

    def allocate_budget(
        self,
        items: Sequence[Any],
        budget: int,
        strategy: AllocationStrategy = AllocationStrategy.PRIORITY_FIRST,
    ) -> AllocationResult:
        """Allocate token budget across content items.

        Distributes the available budget across items based on the specified
        strategy. Higher-priority items (priority=1) are favored when budget
        is limited.

        Args:
            items: Sequence of content items implementing ContentItem protocol.
                Each must have id, content, and priority attributes.
            budget: Total token budget available for allocation
            strategy: Strategy for distributing budget across items

        Returns:
            AllocationResult with allocated items, metrics, and dropped IDs

        Raises:
            ValueError: If budget is not positive

        Example:
            result = manager.allocate_budget(
                items=sources,
                budget=100_000,
                strategy=AllocationStrategy.PRIORITY_FIRST,
            )
        """
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")

        if not items:
            return AllocationResult(
                items=[],
                tokens_used=0,
                tokens_available=budget,
                fidelity=1.0,
                warnings=[],
                dropped_ids=[],
            )

        # Sort items by priority
        sorted_items = self._sort_by_priority(items)

        # Estimate tokens for all items
        item_tokens: list[tuple[Any, int]] = []
        total_original_tokens = 0
        for item in sorted_items:
            tokens = self._get_item_tokens(item)
            item_tokens.append((item, tokens))
            total_original_tokens += tokens

        # Dispatch to strategy-specific allocation
        if strategy == AllocationStrategy.PRIORITY_FIRST:
            return self._allocate_priority_first(item_tokens, budget, total_original_tokens)
        elif strategy == AllocationStrategy.EQUAL_SHARE:
            return self._allocate_equal_share(item_tokens, budget, total_original_tokens)
        else:  # strategy == AllocationStrategy.PROPORTIONAL
            return self._allocate_proportional(item_tokens, budget, total_original_tokens)

    def _allocate_priority_first(
        self,
        item_tokens: list[tuple[Any, int]],
        budget: int,
        total_original_tokens: int,
    ) -> AllocationResult:
        """Allocate budget to highest-priority items first.

        Items are allocated in priority order. Each item gets its full
        token requirement if budget allows, otherwise it's either allocated
        remaining budget (needs_summarization=True) or dropped.

        Args:
            item_tokens: List of (item, token_count) tuples, sorted by priority
            budget: Total budget available
            total_original_tokens: Sum of all original token counts

        Returns:
            AllocationResult with allocation details
        """
        allocated_items: list[AllocatedItem] = []
        dropped_ids: list[str] = []
        warnings: list[str] = []
        remaining_budget = budget
        total_allocated_tokens = 0

        for item, tokens in item_tokens:
            item_id = getattr(item, "id", str(id(item)))
            item_priority = getattr(item, "priority", 999)
            item_content = getattr(item, "content", "")
            item_protected = getattr(item, "protected", False)

            if remaining_budget <= 0:
                if item_protected:
                    # Protected items must be allocated even without budget
                    # They will need aggressive summarization
                    allocated_items.append(
                        AllocatedItem(
                            id=item_id,
                            content=item_content,
                            priority=item_priority,
                            original_tokens=tokens,
                            allocated_tokens=1,  # Minimum allocation
                            needs_summarization=True,
                        )
                    )
                    total_allocated_tokens += 1
                    warnings.append(
                        f"Protected item {item_id} force-allocated with minimal budget: "
                        f"{tokens} tokens -> 1 allocated (needs aggressive summarization)"
                    )
                else:
                    # No budget left - drop non-protected items
                    dropped_ids.append(item_id)
                    warnings.append(f"Dropped item {item_id} (priority={item_priority}): no budget remaining")
                continue

            if tokens <= remaining_budget:
                # Full allocation
                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=tokens,
                        needs_summarization=False,
                    )
                )
                remaining_budget -= tokens
                total_allocated_tokens += tokens
            else:
                # Partial allocation - needs summarization
                allocated_tokens = remaining_budget
                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=allocated_tokens,
                        needs_summarization=True,
                    )
                )
                remaining_budget = 0
                total_allocated_tokens += allocated_tokens
                warnings.append(f"Item {item_id} needs summarization: {tokens} tokens -> {allocated_tokens} allocated")

        # Calculate fidelity
        fidelity = self._calculate_fidelity(allocated_items, total_original_tokens)

        logger.debug(
            f"Priority-first allocation: {len(allocated_items)} items allocated, "
            f"{len(dropped_ids)} dropped, fidelity={fidelity:.2%}"
        )

        return AllocationResult(
            items=allocated_items,
            tokens_used=total_allocated_tokens,
            tokens_available=budget,
            fidelity=fidelity,
            warnings=warnings,
            dropped_ids=dropped_ids,
        )

    def _allocate_equal_share(
        self,
        item_tokens: list[tuple[Any, int]],
        budget: int,
        total_original_tokens: int,
    ) -> AllocationResult:
        """Allocate budget equally across all items.

        Each item receives budget / num_items tokens. Items requiring
        less than their share get their actual requirement; excess is
        redistributed to items needing more.

        Args:
            item_tokens: List of (item, token_count) tuples, sorted by priority
            budget: Total budget available
            total_original_tokens: Sum of all original token counts

        Returns:
            AllocationResult with allocation details
        """
        if not item_tokens:
            return AllocationResult(
                tokens_available=budget,
                fidelity=1.0,
            )

        num_items = len(item_tokens)
        base_share = budget // num_items

        allocated_items: list[AllocatedItem] = []
        warnings: list[str] = []
        total_allocated_tokens = 0

        # First pass: allocate base share or less
        excess_budget = 0
        items_needing_more: list[tuple[int, Any, int]] = []  # (index, item, tokens)

        for idx, (item, tokens) in enumerate(item_tokens):
            if tokens <= base_share:
                # Item fits in base share
                item_id = getattr(item, "id", str(id(item)))
                item_priority = getattr(item, "priority", 999)
                item_content = getattr(item, "content", "")

                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=tokens,
                        needs_summarization=False,
                    )
                )
                total_allocated_tokens += tokens
                excess_budget += base_share - tokens
            else:
                # Item needs more than base share
                items_needing_more.append((idx, item, tokens))

        # Second pass: redistribute excess to items needing more
        if items_needing_more and excess_budget > 0:
            extra_per_item = excess_budget // len(items_needing_more)
        else:
            extra_per_item = 0

        for _idx, item, tokens in items_needing_more:
            item_id = getattr(item, "id", str(id(item)))
            item_priority = getattr(item, "priority", 999)
            item_content = getattr(item, "content", "")

            allocated = min(tokens, base_share + extra_per_item)
            needs_summarization = allocated < tokens

            allocated_items.append(
                AllocatedItem(
                    id=item_id,
                    content=item_content,
                    priority=item_priority,
                    original_tokens=tokens,
                    allocated_tokens=allocated,
                    needs_summarization=needs_summarization,
                )
            )
            total_allocated_tokens += allocated

            if needs_summarization:
                warnings.append(
                    f"Item {item_id} needs summarization: {tokens} tokens -> {allocated} allocated (equal share)"
                )

        # Re-sort by priority for consistent output
        allocated_items.sort(key=lambda x: x.priority)

        # Calculate fidelity
        fidelity = self._calculate_fidelity(allocated_items, total_original_tokens)

        logger.debug(
            f"Equal-share allocation: {len(allocated_items)} items, base share={base_share}, fidelity={fidelity:.2%}"
        )

        return AllocationResult(
            items=allocated_items,
            tokens_used=total_allocated_tokens,
            tokens_available=budget,
            fidelity=fidelity,
            warnings=warnings,
            dropped_ids=[],  # Equal share doesn't drop items
        )

    def _allocate_proportional(
        self,
        item_tokens: list[tuple[Any, int]],
        budget: int,
        total_original_tokens: int,
    ) -> AllocationResult:
        """Allocate budget proportional to item sizes.

        Each item receives budget * (item_tokens / total_tokens).
        Larger items get proportionally larger allocations.

        Args:
            item_tokens: List of (item, token_count) tuples, sorted by priority
            budget: Total budget available
            total_original_tokens: Sum of all original token counts

        Returns:
            AllocationResult with allocation details
        """
        if not item_tokens:
            return AllocationResult(
                tokens_available=budget,
                fidelity=1.0,
            )

        # If total fits in budget, no compression needed
        if total_original_tokens <= budget:
            allocated_items: list[AllocatedItem] = []
            for item, tokens in item_tokens:
                item_id = getattr(item, "id", str(id(item)))
                item_priority = getattr(item, "priority", 999)
                item_content = getattr(item, "content", "")

                allocated_items.append(
                    AllocatedItem(
                        id=item_id,
                        content=item_content,
                        priority=item_priority,
                        original_tokens=tokens,
                        allocated_tokens=tokens,
                        needs_summarization=False,
                    )
                )

            return AllocationResult(
                items=allocated_items,
                tokens_used=total_original_tokens,
                tokens_available=budget,
                fidelity=1.0,
                warnings=[],
                dropped_ids=[],
            )

        # Proportional allocation with compression
        compression_ratio = budget / total_original_tokens
        allocated_items = []
        warnings: list[str] = []
        total_allocated_tokens = 0

        for item, tokens in item_tokens:
            item_id = getattr(item, "id", str(id(item)))
            item_priority = getattr(item, "priority", 999)
            item_content = getattr(item, "content", "")

            # Allocate proportionally, minimum 1 token
            allocated = max(1, int(tokens * compression_ratio))

            allocated_items.append(
                AllocatedItem(
                    id=item_id,
                    content=item_content,
                    priority=item_priority,
                    original_tokens=tokens,
                    allocated_tokens=allocated,
                    needs_summarization=allocated < tokens,
                )
            )
            total_allocated_tokens += allocated

            if allocated < tokens:
                warnings.append(
                    f"Item {item_id} compressed: {tokens} -> {allocated} tokens ({compression_ratio:.1%} of original)"
                )

        # Calculate fidelity
        fidelity = self._calculate_fidelity(allocated_items, total_original_tokens)

        logger.debug(
            f"Proportional allocation: {len(allocated_items)} items, "
            f"compression={compression_ratio:.2%}, fidelity={fidelity:.2%}"
        )

        return AllocationResult(
            items=allocated_items,
            tokens_used=total_allocated_tokens,
            tokens_available=budget,
            fidelity=fidelity,
            warnings=warnings,
            dropped_ids=[],  # Proportional doesn't drop items
        )

    def _calculate_fidelity(
        self,
        allocated_items: list[AllocatedItem],
        total_original_tokens: int,
    ) -> float:
        """Calculate overall fidelity score for an allocation.

        Fidelity represents how much of the original content is preserved:
        - 1.0 = All items allocated at full fidelity
        - 0.0 = All content dropped or maximally compressed

        Dropped items are implicitly accounted for since they contribute
        0 to the allocated token total.

        Args:
            allocated_items: Items that received allocation
            total_original_tokens: Total tokens in original content

        Returns:
            Fidelity score from 0.0 to 1.0
        """
        if total_original_tokens <= 0:
            return 1.0

        # Sum of allocated tokens represents preserved content
        total_allocated = sum(item.allocated_tokens for item in allocated_items)

        # Fidelity is ratio of allocated to original
        fidelity = total_allocated / total_original_tokens

        # Clamp to valid range
        return max(0.0, min(1.0, fidelity))
