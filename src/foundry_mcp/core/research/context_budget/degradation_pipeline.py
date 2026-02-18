"""Degradation pipeline for graceful content compression.

Implements a centralized fallback chain for progressively degrading content
to fit within token budget constraints.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from foundry_mcp.core.errors.research import ProtectedContentOverflowError

from .constants import (
    CHARS_PER_TOKEN,
    CONDENSED_MIN_FIDELITY,
    HEADLINE_MIN_FIDELITY,
    MIN_ITEMS_PER_PHASE,
    TOP_PRIORITY_ITEMS,
    TRUNCATION_MARKER,
)
from .degradation_models import (
    ChunkFailure,
    ChunkResult,
    DegradationLevel,
    DegradationResult,
    DegradationStep,
)
from .models import AllocatedItem, ContentItem

logger = logging.getLogger(__name__)


class DegradationPipeline:
    """Centralized fallback chain for graceful content degradation.

    Implements the degradation chain:
    FULL -> KEY_POINTS -> HEADLINE -> TRUNCATE -> DROP

    The pipeline progressively degrades content to fit within budget:
    1. Start with full content
    2. If over budget, summarize to KEY_POINTS (~30%)
    3. If still over, summarize to HEADLINE (~10%)
    4. If still over, TRUNCATE with warning (always enabled)
    5. If still over and allow_content_dropping=True, DROP lowest priority

    Guardrails:
    - Protected items are never dropped
    - Top-5 priority items never go below condensed fidelity (~30%)
    - Min 3 items per phase preserved when possible
    - Truncation fallback is always enabled (hardcoded)

    Warning Codes:
    - PRIORITY_SUMMARIZED: A top-priority item was degraded (summarized/truncated)
    - CONTENT_DROPPED: A low-priority item was dropped
    - CONTENT_TRUNCATED: Content was truncated
    - PROTECTED_OVERFLOW: Protected item force-allocated with minimal budget
    - TOKEN_BUDGET_FLOORED: Item preserved due to min items guardrail

    Example:
        pipeline = DegradationPipeline(
            allow_content_dropping=True,
            min_items=3,
            priority_items=5,
        )
        result = pipeline.degrade(
            items=sources,
            budget=50_000,
        )
        if result.warnings:
            print(f"Degradation warnings: {result.warnings}")
    """

    def __init__(
        self,
        *,
        token_estimator: Optional[Callable[[str], int]] = None,
        allow_content_dropping: bool = False,
        min_items: int = MIN_ITEMS_PER_PHASE,
        priority_items: int = TOP_PRIORITY_ITEMS,
    ):
        """Initialize the degradation pipeline.

        Args:
            token_estimator: Custom function to estimate tokens.
                If not provided, uses heuristic (len/4).
            allow_content_dropping: If True, allows dropping lowest-priority
                items when other degradation levels fail. Default False.
            min_items: Minimum items to preserve per phase (guardrail).
                Default is MIN_ITEMS_PER_PHASE (3).
            priority_items: Number of top-priority items to preserve at
                minimum condensed fidelity. Default is TOP_PRIORITY_ITEMS (5).
        """
        self._token_estimator = token_estimator
        self._allow_content_dropping = allow_content_dropping
        self._min_items = min_items
        self._priority_items = priority_items

    def _estimate_tokens(self, content: str) -> int:
        """Estimate tokens for content."""
        if self._token_estimator:
            return self._token_estimator(content)
        return max(1, len(content) // CHARS_PER_TOKEN)

    def _truncate_content(self, content: str, target_tokens: int) -> str:
        """Truncate content to fit target token budget.

        Args:
            content: Original content
            target_tokens: Target token count

        Returns:
            Truncated content with marker
        """
        if target_tokens <= 0:
            return TRUNCATION_MARKER.strip()

        # Reserve space for truncation marker
        marker_tokens = len(TRUNCATION_MARKER) // CHARS_PER_TOKEN + 1
        content_tokens = max(1, target_tokens - marker_tokens)
        content_chars = content_tokens * CHARS_PER_TOKEN

        if len(content) <= content_chars:
            return content

        return content[:content_chars].rstrip() + TRUNCATION_MARKER

    def _is_priority_item(self, item_index: int) -> bool:
        """Check if an item is in the top priority set.

        Args:
            item_index: Zero-based index in priority-sorted list

        Returns:
            True if item is in top priority_items (default 5)
        """
        return item_index < self._priority_items

    def _get_min_priority_allocation(self, original_tokens: int) -> int:
        """Get minimum token allocation for priority items.

        Priority items must maintain at least condensed fidelity (30%).

        Args:
            original_tokens: Original token count

        Returns:
            Minimum tokens to allocate (at least 30% of original)
        """
        return max(1, int(original_tokens * CONDENSED_MIN_FIDELITY))

    def _get_headline_allocation(self, original_tokens: int) -> int:
        """Get headline-level token allocation for protected items.

        Headline is the most aggressive compression (~10% of original).
        Used as last resort for protected content overflow.

        Args:
            original_tokens: Original token count

        Returns:
            Minimum tokens for headline level (at least 10% of original)
        """
        return max(1, int(original_tokens * HEADLINE_MIN_FIDELITY))

    def _check_protected_content_budget(
        self,
        protected_items: Sequence[ContentItem],
        budget: int,
    ) -> tuple[bool, int, list[str]]:
        """Check if protected content fits within budget at headline level.

        Args:
            protected_items: List of protected content items
            budget: Available token budget

        Returns:
            Tuple of (fits, total_headline_tokens, item_ids)
        """
        total_headline_tokens = 0
        item_ids = []

        for item in protected_items:
            item_tokens = self._estimate_tokens(item.content)
            headline_tokens = self._get_headline_allocation(item_tokens)
            total_headline_tokens += headline_tokens
            item_ids.append(item.id)

        return (total_headline_tokens <= budget, total_headline_tokens, item_ids)

    def _emit_chunk_warning(
        self,
        item_id: str,
        chunk_id: str,
        message: str,
        *,
        level: Optional[DegradationLevel] = None,
        tokens: Optional[int] = None,
    ) -> str:
        """Generate a standardized chunk-level warning message.

        Creates warning messages that include both item_id and chunk_id
        for precise identification of chunk-level issues.

        Args:
            item_id: ID of the parent item
            chunk_id: ID of the specific chunk (e.g., "chunk-0")
            message: Warning message type/description
            level: Optional degradation level for context
            tokens: Optional token count for context

        Returns:
            Formatted warning string
        """
        parts = [f"CHUNK_FAILURE: {message}"]
        parts.append(f"item_id={item_id}")
        parts.append(f"chunk_id={chunk_id}")

        if level is not None:
            parts.append(f"level={level.value}")
        if tokens is not None:
            parts.append(f"tokens={tokens}")

        return " | ".join(parts)

    def _retry_chunk_at_tighter_level(
        self,
        content: str,
        item_id: str,
        chunk_id: str,
        current_level: DegradationLevel,
        target_tokens: int,
    ) -> ChunkResult:
        """Retry a failed chunk at a more aggressive summarization level.

        Attempts to process a chunk that failed at the current level by
        using a tighter degradation level. Progresses through levels until
        success or reaching TRUNCATE as a last resort.

        Args:
            content: Chunk content to process
            item_id: ID of the parent item
            chunk_id: ID of the chunk (e.g., "chunk-0")
            current_level: Level at which the chunk failed
            target_tokens: Target token count for the output

        Returns:
            ChunkResult with processed content and failure history
        """
        failures: list[ChunkFailure] = []
        level = current_level

        while True:
            next_level = level.next_level()

            if next_level is None or next_level == DegradationLevel.DROP:
                # Reached end of chain - use truncation as last resort
                truncated_content = self._truncate_content(content, target_tokens)
                truncated_tokens = self._estimate_tokens(truncated_content)

                return ChunkResult(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    content=truncated_content,
                    tokens=truncated_tokens,
                    level=DegradationLevel.TRUNCATE,
                    success=True,
                    retried=True,
                    failures=failures,
                )

            level = next_level

            # Try the next level
            # For sync pipeline, we use truncation at progressively tighter ratios
            if level == DegradationLevel.KEY_POINTS:
                allocation = self._get_min_priority_allocation(len(content) // CHARS_PER_TOKEN)
            elif level == DegradationLevel.HEADLINE:
                allocation = self._get_headline_allocation(len(content) // CHARS_PER_TOKEN)
            else:
                allocation = target_tokens

            try:
                truncated_content = self._truncate_content(content, allocation)
                truncated_tokens = self._estimate_tokens(truncated_content)

                if truncated_tokens <= target_tokens:
                    return ChunkResult(
                        item_id=item_id,
                        chunk_id=chunk_id,
                        content=truncated_content,
                        tokens=truncated_tokens,
                        level=level,
                        success=True,
                        retried=True,
                        failures=failures,
                    )

                # Still too large, record failure and continue
                failures.append(ChunkFailure(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    original_level=current_level,
                    retry_level=level,
                    error=f"Still exceeds target: {truncated_tokens} > {target_tokens}",
                    recovered=False,
                ))

            except Exception as e:
                # Record the failure and continue to next level
                failures.append(ChunkFailure(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    original_level=current_level,
                    retry_level=level,
                    error=str(e),
                    recovered=False,
                ))

    def _process_chunk_with_retry(
        self,
        content: str,
        item_id: str,
        chunk_id: str,
        target_tokens: int,
        initial_level: DegradationLevel = DegradationLevel.FULL,
    ) -> ChunkResult:
        """Process a single chunk with automatic retry on failure.

        Attempts to process a chunk at the initial level. If processing
        fails or the result exceeds the target, retries at progressively
        tighter levels until success.

        Successful chunk summaries are preserved; only failed chunks are
        retried. This enables partial results when some chunks succeed.

        Args:
            content: Chunk content to process
            item_id: ID of the parent item
            chunk_id: ID of the chunk (e.g., "chunk-0")
            target_tokens: Target token count for the output
            initial_level: Starting degradation level

        Returns:
            ChunkResult with processed content and any failures
        """
        chunk_tokens = self._estimate_tokens(content)

        # If content already fits, return as-is
        if chunk_tokens <= target_tokens:
            return ChunkResult(
                item_id=item_id,
                chunk_id=chunk_id,
                content=content,
                tokens=chunk_tokens,
                level=initial_level,
                success=True,
                retried=False,
                failures=[],
            )

        # Content doesn't fit - try truncation at current level first
        try:
            truncated_content = self._truncate_content(content, target_tokens)
            truncated_tokens = self._estimate_tokens(truncated_content)

            if truncated_tokens <= target_tokens:
                return ChunkResult(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    content=truncated_content,
                    tokens=truncated_tokens,
                    level=initial_level,
                    success=True,
                    retried=False,
                    failures=[],
                )
        except Exception as e:
            # Initial truncation failed - record and retry at tighter level
            logger.warning(
                f"Chunk truncation failed for {item_id}/{chunk_id}: {e}"
            )

        # Retry at tighter levels
        return self._retry_chunk_at_tighter_level(
            content=content,
            item_id=item_id,
            chunk_id=chunk_id,
            current_level=initial_level,
            target_tokens=target_tokens,
        )

    def process_chunked_item(
        self,
        item_id: str,
        chunks: list[str],
        target_tokens_per_chunk: int,
        initial_level: DegradationLevel = DegradationLevel.FULL,
    ) -> tuple[list[ChunkResult], list[str]]:
        """Process multiple chunks for a single item with failure handling.

        Processes each chunk with automatic retry on failure. Preserves
        successful chunk summaries and retries failed chunks at tighter
        levels. Returns warnings with item_id and chunk_id for each issue.

        Args:
            item_id: ID of the parent item
            chunks: List of chunk content strings
            target_tokens_per_chunk: Target tokens per chunk
            initial_level: Starting degradation level for all chunks

        Returns:
            Tuple of (chunk_results, warnings) where:
            - chunk_results: List of ChunkResult for each chunk
            - warnings: List of warning messages with item_id and chunk_id
        """
        results: list[ChunkResult] = []
        warnings: list[str] = []

        for i, chunk_content in enumerate(chunks):
            chunk_id = f"chunk-{i}"

            result = self._process_chunk_with_retry(
                content=chunk_content,
                item_id=item_id,
                chunk_id=chunk_id,
                target_tokens=target_tokens_per_chunk,
                initial_level=initial_level,
            )

            results.append(result)

            # Generate warnings for any failures
            if result.failures:
                for failure in result.failures:
                    warning = self._emit_chunk_warning(
                        item_id=failure.item_id,
                        chunk_id=failure.chunk_id,
                        message=f"Retry at {failure.retry_level.value if failure.retry_level else 'unknown'}: {failure.error}",
                        level=failure.original_level,
                    )
                    warnings.append(warning)

            # Warn if chunk was retried at tighter level
            if result.retried:
                warning = self._emit_chunk_warning(
                    item_id=item_id,
                    chunk_id=chunk_id,
                    message=f"Recovered at {result.level.value}",
                    level=result.level,
                    tokens=result.tokens,
                )
                warnings.append(warning)

        return results, warnings

    def degrade(
        self,
        items: Sequence[ContentItem],
        budget: int,
    ) -> DegradationResult:
        """Run the degradation pipeline on items to fit budget.

        Attempts progressive degradation to fit content within budget:
        1. Allocate items at full fidelity (priority order)
        2. For items that don't fit, try KEY_POINTS summarization
        3. If still over, try HEADLINE summarization
        4. If still over, TRUNCATE (always enabled)
        5. If still over and allow_content_dropping=True, DROP

        Protected content handling:
        - Protected items are never dropped
        - If budget is tight, protected items get headline allocation (~10%)
        - If protected content exceeds budget even at headline level,
          raises ProtectedContentOverflowError with remediation guidance

        Args:
            items: Content items to degrade (must have id, content, priority)
            budget: Total token budget available

        Returns:
            DegradationResult with degraded items and metadata

        Raises:
            ValueError: If budget is not positive
            ProtectedContentOverflowError: If protected content exceeds budget
                even at headline level
        """
        if not items:
            return DegradationResult(fidelity=1.0)

        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")

        # Pre-check: Verify protected content fits at headline level
        protected_items_list = [i for i in items if i.protected]
        if protected_items_list:
            fits, headline_tokens, protected_ids = self._check_protected_content_budget(
                protected_items_list, budget
            )
            if not fits:
                raise ProtectedContentOverflowError(
                    protected_tokens=headline_tokens,
                    budget=budget,
                    item_ids=protected_ids,
                )

        # Sort by priority (1 = highest, first)
        sorted_items = sorted(items, key=lambda x: x.priority)

        # Track state
        allocated: list[AllocatedItem] = []
        steps: list[DegradationStep] = []
        dropped_ids: list[str] = []
        warnings: list[str] = []
        remaining_budget = budget
        total_original_tokens = 0
        min_items_enforced = False

        # Count protected and non-protected items
        protected_items = [i for i in sorted_items if i.protected]
        droppable_items = [i for i in sorted_items if not i.protected]

        for item_index, item in enumerate(sorted_items):
            is_priority = self._is_priority_item(item_index)
            item_tokens = self._estimate_tokens(item.content)
            total_original_tokens += item_tokens

            # Check if item fits at full fidelity
            if item_tokens <= remaining_budget:
                # Full fidelity allocation
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=item.content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=item_tokens,
                    needs_summarization=False,
                ))
                remaining_budget -= item_tokens
                continue

            # Item doesn't fit at full fidelity - use truncation fallback
            # Note: KEY_POINTS and HEADLINE summarization require async operations
            # and would be handled by ContentSummarizer. The sync pipeline uses
            # truncation as the fallback (always enabled per spec).

            if remaining_budget > 0:
                # For priority items, enforce minimum condensed fidelity (30%)
                if is_priority:
                    min_allocation = self._get_min_priority_allocation(item_tokens)
                    target_tokens = max(remaining_budget, min_allocation)
                else:
                    target_tokens = remaining_budget

                # Truncate to fit target budget
                truncated_content = self._truncate_content(item.content, target_tokens)
                truncated_tokens = self._estimate_tokens(truncated_content)

                # Determine the degradation level
                allocation_ratio = truncated_tokens / item_tokens if item_tokens > 0 else 1.0
                if allocation_ratio >= CONDENSED_MIN_FIDELITY:
                    to_level = DegradationLevel.KEY_POINTS
                else:
                    to_level = DegradationLevel.TRUNCATE

                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.FULL,
                    to_level=to_level,
                    original_tokens=item_tokens,
                    result_tokens=truncated_tokens,
                    success=True,
                    warning=f"Content degraded from {item_tokens} to {truncated_tokens} tokens",
                ))

                allocated.append(AllocatedItem(
                    id=item.id,
                    content=truncated_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=truncated_tokens,
                    needs_summarization=True,  # Mark as degraded
                ))
                remaining_budget -= truncated_tokens

                # Emit appropriate warning based on priority status
                if is_priority:
                    warnings.append(
                        f"PRIORITY_SUMMARIZED: Priority item {item.id} degraded from "
                        f"{item_tokens} to {truncated_tokens} tokens "
                        f"(fidelity={allocation_ratio:.1%}, min={CONDENSED_MIN_FIDELITY:.0%})"
                    )
                else:
                    warnings.append(
                        f"CONTENT_TRUNCATED: Item {item.id} truncated from "
                        f"{item_tokens} to {truncated_tokens} tokens"
                    )
                continue

            # No budget remaining - consider dropping
            # Protected items and priority items are never dropped
            if item.protected:
                # Protected items get headline allocation (~10%) as last resort
                # (pre-check guarantees this fits within budget)
                headline_allocation = self._get_headline_allocation(item_tokens)
                headline_content = self._truncate_content(item.content, headline_allocation)
                headline_tokens = self._estimate_tokens(headline_content)

                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.FULL,
                    to_level=DegradationLevel.HEADLINE,
                    original_tokens=item_tokens,
                    result_tokens=headline_tokens,
                    success=True,
                    warning="Protected item compressed to headline level",
                ))
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=headline_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=headline_tokens,
                    needs_summarization=True,
                ))
                warnings.append(
                    f"PROTECTED_OVERFLOW: Protected item {item.id} compressed to headline "
                    f"({headline_tokens}/{item_tokens} tokens, "
                    f"fidelity={headline_tokens/item_tokens:.1%})"
                )
                continue

            # Priority items (top-5) must maintain at least condensed fidelity
            if is_priority:
                min_allocation = self._get_min_priority_allocation(item_tokens)
                minimal_content = self._truncate_content(item.content, min_allocation)
                minimal_tokens = self._estimate_tokens(minimal_content)
                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.FULL,
                    to_level=DegradationLevel.KEY_POINTS,
                    original_tokens=item_tokens,
                    result_tokens=minimal_tokens,
                    success=False,
                    warning="Priority item force-allocated at condensed fidelity",
                ))
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=minimal_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=minimal_tokens,
                    needs_summarization=True,
                ))
                warnings.append(
                    f"PRIORITY_SUMMARIZED: Priority item {item.id} force-allocated "
                    f"at condensed fidelity ({minimal_tokens}/{item_tokens} tokens)"
                )
                continue

            # Check if we can drop this low-priority item
            if self._allow_content_dropping:
                # Check min items guardrail
                current_allocated_count = len(allocated) + len(protected_items) - len([
                    a for a in allocated if any(p.id == a.id for p in protected_items)
                ])
                # Count remaining items that could still be allocated
                remaining_droppable = len([
                    d for d in droppable_items
                    if d.id not in dropped_ids and d.id != item.id
                ])
                potential_total = current_allocated_count + remaining_droppable

                if potential_total >= self._min_items:
                    # Safe to drop
                    steps.append(DegradationStep(
                        item_id=item.id,
                        from_level=DegradationLevel.TRUNCATE,
                        to_level=DegradationLevel.DROP,
                        original_tokens=item_tokens,
                        result_tokens=0,
                        success=True,
                    ))
                    dropped_ids.append(item.id)
                    warnings.append(
                        f"CONTENT_DROPPED: Item {item.id} dropped "
                        f"(priority={item.priority}, tokens={item_tokens})"
                    )
                else:
                    # Would violate min items - force allocate with truncation
                    min_items_enforced = True
                    minimal_content = self._truncate_content(item.content, 1)
                    steps.append(DegradationStep(
                        item_id=item.id,
                        from_level=DegradationLevel.DROP,
                        to_level=DegradationLevel.TRUNCATE,
                        original_tokens=item_tokens,
                        result_tokens=1,
                        success=False,
                        warning=f"Min items guardrail ({self._min_items}) prevented drop",
                    ))
                    allocated.append(AllocatedItem(
                        id=item.id,
                        content=minimal_content,
                        priority=item.priority,
                        original_tokens=item_tokens,
                        allocated_tokens=1,
                        needs_summarization=True,
                    ))
                    warnings.append(
                        f"TOKEN_BUDGET_FLOORED: Item {item.id} preserved due to "
                        f"min items guardrail ({self._min_items} items)"
                    )
            else:
                # Dropping not allowed - force allocate with minimal truncation
                minimal_content = self._truncate_content(item.content, 1)
                steps.append(DegradationStep(
                    item_id=item.id,
                    from_level=DegradationLevel.TRUNCATE,
                    to_level=DegradationLevel.TRUNCATE,
                    original_tokens=item_tokens,
                    result_tokens=1,
                    success=False,
                    warning="Content dropping disabled, forced minimal allocation",
                ))
                allocated.append(AllocatedItem(
                    id=item.id,
                    content=minimal_content,
                    priority=item.priority,
                    original_tokens=item_tokens,
                    allocated_tokens=1,
                    needs_summarization=True,
                ))
                warnings.append(
                    f"CONTENT_TRUNCATED: Item {item.id} force-allocated with "
                    f"minimal budget (content_dropping=False)"
                )

        # Calculate fidelity
        total_allocated = sum(item.allocated_tokens for item in allocated)
        fidelity = total_allocated / total_original_tokens if total_original_tokens > 0 else 1.0

        return DegradationResult(
            items=allocated,
            tokens_used=total_allocated,
            fidelity=max(0.0, min(1.0, fidelity)),
            steps=steps,
            dropped_ids=dropped_ids,
            warnings=warnings,
            min_items_enforced=min_items_enforced,
        )
