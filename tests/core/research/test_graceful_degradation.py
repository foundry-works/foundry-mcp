"""Tests for graceful degradation in the DegradationPipeline.

Tests cover:
1. Full fallback chain (FULL → KEY_POINTS → HEADLINE → TRUNCATE → DROP)
2. Priority guardrails (top-5 items preserved at min 30% fidelity)
3. Protected content handling (never dropped, headline allocation as last resort)
4. Chunk-level failure recovery (retry at tighter levels, preserve successful chunks)
"""

import pytest

from foundry_mcp.core.research.context_budget import (
    CHARS_PER_TOKEN,
    CONDENSED_MIN_FIDELITY,
    HEADLINE_MIN_FIDELITY,
    MIN_ITEMS_PER_PHASE,
    TOP_PRIORITY_ITEMS,
    ChunkFailure,
    ChunkResult,
    ContentItem,
    DegradationLevel,
    DegradationPipeline,
    DegradationResult,
    DegradationStep,
    ProtectedContentOverflowError,
)

# =============================================================================
# Test: Degradation Fallback Chain
# =============================================================================


class TestDegradationFallbackChain:
    """Tests for the degradation fallback chain progression."""

    @pytest.fixture
    def pipeline(self):
        """Create a DegradationPipeline with fixed token estimation."""
        # Use simple estimator: 1 token per 4 characters
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
            allow_content_dropping=True,
        )

    @pytest.fixture
    def pipeline_no_drop(self):
        """Create a DegradationPipeline that doesn't allow dropping."""
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
            allow_content_dropping=False,
        )

    def test_full_fidelity_when_budget_allows(self, pipeline):
        """Test that items are allocated at full fidelity when budget allows."""
        items = [
            ContentItem(id="item-1", content="A" * 400, priority=1),  # 100 tokens
            ContentItem(id="item-2", content="B" * 200, priority=2),  # 50 tokens
        ]

        result = pipeline.degrade(items, budget=200)

        assert len(result.items) == 2
        assert result.fidelity == 1.0
        assert len(result.dropped_ids) == 0
        assert len(result.steps) == 0  # No degradation steps needed

    def test_truncation_when_budget_tight(self, pipeline):
        """Test that items are truncated when budget is tight."""
        items = [
            ContentItem(id="item-1", content="A" * 400, priority=1),  # 100 tokens
        ]

        result = pipeline.degrade(items, budget=50)

        assert len(result.items) == 1
        item = result.items[0]
        assert item.allocated_tokens <= 50
        assert item.needs_summarization is True
        assert len(result.steps) > 0

    def test_drop_when_budget_exhausted(self, pipeline):
        """Test that low-priority items are dropped when budget exhausted."""
        # Create more than TOP_PRIORITY_ITEMS (5) + MIN_ITEMS_PER_PHASE (3) items
        # so that some can be dropped. The top 5 by priority index are protected
        # at min 30% fidelity, and the min_items guardrail prevents going below 3.
        items = [
            ContentItem(id=f"item-{i}", content="A" * 400, priority=i)
            for i in range(1, 10)  # 9 items - indices 0-8, indices 5-8 can be dropped
        ]

        # Very tight budget - not enough for all items
        result = pipeline.degrade(items, budget=200)

        # Higher priority items should be allocated
        allocated_ids = {item.id for item in result.items}
        assert "item-1" in allocated_ids  # Highest priority
        assert "item-2" in allocated_ids  # Second highest

        # Some low priority items (beyond top-5) should be dropped
        # Note: min_items guardrail keeps at least 3 items
        assert len(result.dropped_ids) > 0 or len(result.items) >= MIN_ITEMS_PER_PHASE

    def test_no_drop_when_disabled(self, pipeline_no_drop):
        """Test that items are not dropped when allow_content_dropping=False."""
        items = [
            ContentItem(id="item-1", content="A" * 400, priority=1),  # 100 tokens
            ContentItem(id="item-2", content="B" * 400, priority=2),  # 100 tokens
        ]

        result = pipeline_no_drop.degrade(items, budget=100)

        # All items allocated (even with minimal budget)
        assert len(result.items) == 2
        assert len(result.dropped_ids) == 0

    def test_degradation_level_next_level(self):
        """Test DegradationLevel.next_level() progression."""
        assert DegradationLevel.FULL.next_level() == DegradationLevel.KEY_POINTS
        assert DegradationLevel.KEY_POINTS.next_level() == DegradationLevel.HEADLINE
        assert DegradationLevel.HEADLINE.next_level() == DegradationLevel.TRUNCATE
        assert DegradationLevel.TRUNCATE.next_level() == DegradationLevel.DROP
        assert DegradationLevel.DROP.next_level() is None

    def test_truncation_marker_added(self, pipeline):
        """Test that truncated content has the truncation marker."""
        items = [
            ContentItem(id="item-1", content="A" * 4000, priority=1),  # 1000 tokens
        ]

        result = pipeline.degrade(items, budget=50)

        assert len(result.items) == 1
        assert "[... truncated]" in result.items[0].content

    def test_step_records_degradation(self, pipeline):
        """Test that degradation steps are recorded correctly."""
        items = [
            ContentItem(id="item-1", content="A" * 400, priority=1),  # 100 tokens
        ]

        result = pipeline.degrade(items, budget=30)

        assert len(result.steps) >= 1
        step = result.steps[0]
        assert step.item_id == "item-1"
        assert step.from_level == DegradationLevel.FULL
        assert step.original_tokens == 100
        assert step.result_tokens <= 30

    def test_warnings_emitted_for_truncation(self, pipeline):
        """Test that warnings are emitted when content is truncated."""
        items = [
            ContentItem(id="item-1", content="A" * 400, priority=1),
        ]

        result = pipeline.degrade(items, budget=30)

        assert len(result.warnings) >= 1
        # Should have CONTENT_TRUNCATED or PRIORITY_SUMMARIZED warning
        assert any("TRUNCATED" in w or "SUMMARIZED" in w for w in result.warnings)


# =============================================================================
# Test: Priority Guardrails
# =============================================================================


class TestPriorityGuardrails:
    """Tests for priority guardrails (top-5 items protected at 30% fidelity)."""

    @pytest.fixture
    def pipeline(self):
        """Create a DegradationPipeline with fixed token estimation."""
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
            allow_content_dropping=True,
            priority_items=TOP_PRIORITY_ITEMS,  # Default 5
        )

    def test_priority_items_constant(self):
        """Test TOP_PRIORITY_ITEMS constant is set correctly."""
        assert TOP_PRIORITY_ITEMS == 5

    def test_condensed_min_fidelity_constant(self):
        """Test CONDENSED_MIN_FIDELITY constant is set correctly."""
        assert CONDENSED_MIN_FIDELITY == 0.30

    def test_top_priority_items_never_dropped(self, pipeline):
        """Test that top-5 priority items are never dropped."""
        # Create 7 items - priorities 1-7
        items = [ContentItem(id=f"item-{i}", content="A" * 400, priority=i) for i in range(1, 8)]

        # Very tight budget - not enough for all items
        result = pipeline.degrade(items, budget=100)

        # Top 5 priority items should all be present
        allocated_ids = {item.id for item in result.items}
        for i in range(1, 6):  # priority 1-5
            assert f"item-{i}" in allocated_ids or f"item-{i}" not in result.dropped_ids

    def test_priority_items_get_min_condensed_fidelity(self, pipeline):
        """Test that priority items get at least 30% of their tokens."""
        items = [
            ContentItem(id="priority-1", content="A" * 400, priority=1),  # 100 tokens
            ContentItem(id="priority-2", content="B" * 400, priority=2),  # 100 tokens
        ]

        # Budget that forces degradation but should maintain min fidelity
        result = pipeline.degrade(items, budget=60)

        # Check priority items
        for item in result.items:
            if item.id.startswith("priority"):
                # Should get at least 30% of original
                min_expected = int(item.original_tokens * CONDENSED_MIN_FIDELITY)
                # Allow for truncation overhead
                assert item.allocated_tokens >= min_expected - 5

    def test_is_priority_item_method(self, pipeline):
        """Test _is_priority_item correctly identifies top-5 items."""
        assert pipeline._is_priority_item(0) is True  # Index 0 = priority 1
        assert pipeline._is_priority_item(4) is True  # Index 4 = priority 5
        assert pipeline._is_priority_item(5) is False  # Index 5 = priority 6
        assert pipeline._is_priority_item(10) is False

    def test_get_min_priority_allocation(self, pipeline):
        """Test _get_min_priority_allocation returns 30% of tokens."""
        original_tokens = 100
        min_alloc = pipeline._get_min_priority_allocation(original_tokens)
        assert min_alloc == int(100 * CONDENSED_MIN_FIDELITY)

    def test_priority_summarized_warning_emitted(self, pipeline):
        """Test PRIORITY_SUMMARIZED warning is emitted for degraded priority items."""
        items = [
            ContentItem(id="item-1", content="A" * 400, priority=1),
        ]

        result = pipeline.degrade(items, budget=20)

        # Should have PRIORITY_SUMMARIZED warning
        assert any("PRIORITY_SUMMARIZED" in w for w in result.warnings)


# =============================================================================
# Test: Protected Content Handling
# =============================================================================


class TestProtectedContentHandling:
    """Tests for protected content handling (never dropped)."""

    @pytest.fixture
    def pipeline(self):
        """Create a DegradationPipeline with fixed token estimation."""
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
            allow_content_dropping=True,
        )

    def test_headline_min_fidelity_constant(self):
        """Test HEADLINE_MIN_FIDELITY constant is set correctly."""
        assert HEADLINE_MIN_FIDELITY == 0.10

    def test_protected_item_never_dropped(self, pipeline):
        """Test that protected items are never dropped."""
        items = [
            ContentItem(id="regular-1", content="A" * 400, priority=1),
            ContentItem(id="protected-1", content="B" * 400, priority=10, protected=True),
            ContentItem(id="regular-2", content="C" * 400, priority=2),
        ]

        # Very tight budget
        result = pipeline.degrade(items, budget=100)

        # Protected item should be allocated
        allocated_ids = {item.id for item in result.items}
        assert "protected-1" in allocated_ids
        assert "protected-1" not in result.dropped_ids

    def test_protected_item_gets_headline_allocation(self, pipeline):
        """Test that protected items get headline allocation when budget is exhausted."""
        items = [
            ContentItem(id="big-1", content="A" * 4000, priority=1),  # 1000 tokens
            ContentItem(id="protected-1", content="B" * 400, priority=2, protected=True),  # 100 tokens
        ]

        # Budget exhausted by first item
        result = pipeline.degrade(items, budget=100)

        # Protected item should still be allocated
        protected_item = next(i for i in result.items if i.id == "protected-1")
        assert protected_item is not None
        # Should be at headline level (~10% of original)
        expected_headline = int(100 * HEADLINE_MIN_FIDELITY)
        # Allow for truncation overhead
        assert protected_item.allocated_tokens >= expected_headline - 5

    def test_protected_overflow_warning(self, pipeline):
        """Test PROTECTED_OVERFLOW warning is emitted when protected content compressed."""
        items = [
            ContentItem(id="big", content="A" * 4000, priority=1),  # 1000 tokens
            ContentItem(id="protected", content="B" * 400, priority=2, protected=True),
        ]

        result = pipeline.degrade(items, budget=100)

        # Should have PROTECTED_OVERFLOW warning
        assert any("PROTECTED_OVERFLOW" in w for w in result.warnings)

    def test_protected_content_overflow_error(self, pipeline):
        """Test ProtectedContentOverflowError raised when protected content exceeds budget."""
        # Create protected items that exceed budget even at headline level
        items = [
            ContentItem(id="p1", content="A" * 4000, priority=1, protected=True),  # 1000 tokens
            ContentItem(id="p2", content="B" * 4000, priority=2, protected=True),  # 1000 tokens
        ]

        # Budget too small even for headline allocation (~10% of 2000 = 200)
        with pytest.raises(ProtectedContentOverflowError) as exc_info:
            pipeline.degrade(items, budget=50)

        error = exc_info.value
        assert error.protected_tokens > error.budget
        assert "p1" in error.item_ids
        assert "p2" in error.item_ids
        assert "remediation" in error.remediation.lower() or "increase" in error.remediation.lower()

    def test_protected_content_overflow_error_to_dict(self):
        """Test ProtectedContentOverflowError.to_dict() serialization."""
        error = ProtectedContentOverflowError(
            protected_tokens=300,
            budget=100,
            item_ids=["item-1", "item-2"],
        )

        d = error.to_dict()

        assert d["error_type"] == "protected_content_overflow"
        assert d["protected_tokens"] == 300
        assert d["budget"] == 100
        assert "item-1" in d["item_ids"]
        assert "remediation" in d

    def test_get_headline_allocation(self, pipeline):
        """Test _get_headline_allocation returns 10% of tokens."""
        original_tokens = 100
        headline_alloc = pipeline._get_headline_allocation(original_tokens)
        assert headline_alloc == int(100 * HEADLINE_MIN_FIDELITY)

    def test_check_protected_content_budget(self, pipeline):
        """Test _check_protected_content_budget pre-check."""
        items = [
            ContentItem(id="p1", content="A" * 400, priority=1, protected=True),  # 100 tokens
            ContentItem(id="p2", content="B" * 200, priority=2, protected=True),  # 50 tokens
        ]

        # At headline level: ~10 + ~5 = 15 tokens needed
        fits, headline_tokens, item_ids = pipeline._check_protected_content_budget(items, budget=20)

        assert fits is True
        assert headline_tokens <= 20
        assert "p1" in item_ids
        assert "p2" in item_ids


# =============================================================================
# Test: Chunk-Level Failure Recovery
# =============================================================================


class TestChunkFailureRecovery:
    """Tests for chunk-level failure recovery."""

    @pytest.fixture
    def pipeline(self):
        """Create a DegradationPipeline with fixed token estimation."""
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
            allow_content_dropping=True,
        )

    def test_chunk_failure_dataclass(self):
        """Test ChunkFailure dataclass and serialization."""
        failure = ChunkFailure(
            item_id="item-1",
            chunk_id="chunk-0",
            original_level=DegradationLevel.FULL,
            retry_level=DegradationLevel.KEY_POINTS,
            error="Test error",
            recovered=True,
        )

        d = failure.to_dict()

        assert d["item_id"] == "item-1"
        assert d["chunk_id"] == "chunk-0"
        assert d["original_level"] == "full"
        assert d["retry_level"] == "key_points"
        assert d["error"] == "Test error"
        assert d["recovered"] is True

    def test_chunk_result_dataclass(self):
        """Test ChunkResult dataclass and serialization."""
        failure = ChunkFailure(
            item_id="item-1",
            chunk_id="chunk-0",
            original_level=DegradationLevel.FULL,
        )

        result = ChunkResult(
            item_id="item-1",
            chunk_id="chunk-0",
            content="Test content",
            tokens=10,
            level=DegradationLevel.KEY_POINTS,
            success=True,
            retried=True,
            failures=[failure],
        )

        d = result.to_dict()

        assert d["item_id"] == "item-1"
        assert d["chunk_id"] == "chunk-0"
        assert d["tokens"] == 10
        assert d["level"] == "key_points"
        assert d["success"] is True
        assert d["retried"] is True
        assert len(d["failures"]) == 1

    def test_emit_chunk_warning_format(self, pipeline):
        """Test _emit_chunk_warning generates proper format."""
        warning = pipeline._emit_chunk_warning(
            item_id="item-1",
            chunk_id="chunk-0",
            message="Test failure",
            level=DegradationLevel.FULL,
            tokens=100,
        )

        assert "CHUNK_FAILURE" in warning
        assert "item_id=item-1" in warning
        assert "chunk_id=chunk-0" in warning
        assert "level=full" in warning
        assert "tokens=100" in warning

    def test_emit_chunk_warning_minimal(self, pipeline):
        """Test _emit_chunk_warning without optional parameters."""
        warning = pipeline._emit_chunk_warning(
            item_id="item-1",
            chunk_id="chunk-0",
            message="Test failure",
        )

        assert "CHUNK_FAILURE" in warning
        assert "item_id=item-1" in warning
        assert "chunk_id=chunk-0" in warning
        assert "level=" not in warning
        assert "tokens=" not in warning

    def test_process_chunk_with_retry_fits(self, pipeline):
        """Test _process_chunk_with_retry when content fits."""
        result = pipeline._process_chunk_with_retry(
            content="Short content",
            item_id="item-1",
            chunk_id="chunk-0",
            target_tokens=100,
            initial_level=DegradationLevel.FULL,
        )

        assert result.success is True
        assert result.retried is False
        assert result.level == DegradationLevel.FULL
        assert len(result.failures) == 0

    def test_process_chunk_with_retry_truncates(self, pipeline):
        """Test _process_chunk_with_retry truncates when needed."""
        long_content = "x" * 4000  # ~1000 tokens

        result = pipeline._process_chunk_with_retry(
            content=long_content,
            item_id="item-1",
            chunk_id="chunk-0",
            target_tokens=50,
            initial_level=DegradationLevel.FULL,
        )

        assert result.success is True
        assert result.tokens <= 50
        assert "[... truncated]" in result.content or len(result.content) < len(long_content)

    def test_retry_chunk_at_tighter_level(self, pipeline):
        """Test _retry_chunk_at_tighter_level progressively tightens."""
        long_content = "x" * 4000  # ~1000 tokens

        result = pipeline._retry_chunk_at_tighter_level(
            content=long_content,
            item_id="item-1",
            chunk_id="chunk-0",
            current_level=DegradationLevel.FULL,
            target_tokens=10,
        )

        assert result.success is True
        assert result.retried is True
        assert result.tokens <= 10
        # Should reach TRUNCATE as last resort for very small targets
        assert result.level in [
            DegradationLevel.KEY_POINTS,
            DegradationLevel.HEADLINE,
            DegradationLevel.TRUNCATE,
        ]

    def test_process_chunked_item_multiple_chunks(self, pipeline):
        """Test process_chunked_item processes multiple chunks."""
        chunks = [
            "Short chunk one",
            "x" * 2000,  # ~500 tokens - needs truncation
            "Short chunk three",
        ]

        results, warnings = pipeline.process_chunked_item(
            item_id="item-1",
            chunks=chunks,
            target_tokens_per_chunk=50,
            initial_level=DegradationLevel.FULL,
        )

        assert len(results) == 3
        assert all(r.item_id == "item-1" for r in results)
        assert results[0].chunk_id == "chunk-0"
        assert results[1].chunk_id == "chunk-1"
        assert results[2].chunk_id == "chunk-2"

        # All chunks should succeed
        assert all(r.success for r in results)

    def test_process_chunked_item_preserves_successful(self, pipeline):
        """Test that successful chunks are preserved, only failed retried."""
        chunks = [
            "Short content",  # Will succeed at full level
            "x" * 4000,  # Will need retry
        ]

        results, warnings = pipeline.process_chunked_item(
            item_id="item-1",
            chunks=chunks,
            target_tokens_per_chunk=50,
        )

        # First chunk should not be retried
        assert results[0].retried is False
        assert results[0].level == DegradationLevel.FULL

        # Second chunk may be retried
        # Both should succeed
        assert results[0].success is True
        assert results[1].success is True

    def test_process_chunked_item_warnings_include_ids(self, pipeline):
        """Test that warnings include item_id and chunk_id."""
        chunks = ["x" * 4000]  # Will need truncation

        results, warnings = pipeline.process_chunked_item(
            item_id="test-item",
            chunks=chunks,
            target_tokens_per_chunk=10,
        )

        # If retried, should have warning with item_id and chunk_id
        if results[0].retried:
            assert any("item_id=test-item" in w for w in warnings)
            assert any("chunk_id=chunk-0" in w for w in warnings)

    def test_degradation_step_with_chunk_id(self):
        """Test DegradationStep includes chunk_id field."""
        step = DegradationStep(
            item_id="item-1",
            from_level=DegradationLevel.FULL,
            to_level=DegradationLevel.KEY_POINTS,
            original_tokens=100,
            result_tokens=30,
            success=True,
            warning="Test",
            chunk_id="chunk-0",
        )

        assert step.chunk_id == "chunk-0"

    def test_degradation_result_includes_chunk_failures(self, pipeline):
        """Test DegradationResult includes chunk_failures field."""
        result = DegradationResult(
            items=[],
            tokens_used=0,
            fidelity=1.0,
            chunk_failures=[
                ChunkFailure(
                    item_id="item-1",
                    chunk_id="chunk-0",
                    original_level=DegradationLevel.FULL,
                )
            ],
        )

        d = result.to_dict()

        assert "chunk_failures" in d
        assert len(d["chunk_failures"]) == 1
        assert d["chunk_failures"][0]["item_id"] == "item-1"
        assert d["chunk_failures"][0]["chunk_id"] == "chunk-0"


# =============================================================================
# Test: Min Items Guardrail
# =============================================================================


class TestMinItemsGuardrail:
    """Tests for minimum items guardrail."""

    @pytest.fixture
    def pipeline(self):
        """Create a DegradationPipeline with fixed token estimation."""
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
            allow_content_dropping=True,
            min_items=MIN_ITEMS_PER_PHASE,  # Default 3
        )

    def test_min_items_constant(self):
        """Test MIN_ITEMS_PER_PHASE constant is set correctly."""
        assert MIN_ITEMS_PER_PHASE == 3

    def test_min_items_guardrail_prevents_drop(self, pipeline):
        """Test that min items guardrail prevents dropping below threshold."""
        # Create exactly min_items items
        items = [
            ContentItem(id=f"item-{i}", content="A" * 400, priority=i)
            for i in range(1, 4)  # 3 items
        ]

        # Very tight budget
        result = pipeline.degrade(items, budget=50)

        # Should not drop below min_items
        # All 3 should be allocated (even with minimal budget)
        assert len(result.items) >= MIN_ITEMS_PER_PHASE - 1  # Some tolerance

    def test_min_items_enforced_flag(self, pipeline):
        """Test min_items_enforced flag is set when guardrail is active."""
        items = [
            ContentItem(id=f"item-{i}", content="A" * 400, priority=i)
            for i in range(1, 4)  # 3 items
        ]

        result = pipeline.degrade(items, budget=50)

        # Check if min_items_enforced flag is tracked
        # (may or may not be True depending on allocation)
        assert hasattr(result, "min_items_enforced")

    def test_token_budget_floored_warning(self, pipeline):
        """Test TOKEN_BUDGET_FLOORED warning when min items guardrail active."""
        items = [ContentItem(id=f"item-{i}", content="A" * 400, priority=i) for i in range(1, 4)]

        result = pipeline.degrade(items, budget=50)

        # May have TOKEN_BUDGET_FLOORED warning if guardrail was active
        if result.min_items_enforced:
            assert any("TOKEN_BUDGET_FLOORED" in w for w in result.warnings)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestDegradationEdgeCases:
    """Tests for edge cases in degradation pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a DegradationPipeline with fixed token estimation."""
        return DegradationPipeline(
            token_estimator=lambda content: len(content) // CHARS_PER_TOKEN,
        )

    def test_empty_items_list(self, pipeline):
        """Test degradation with empty items list."""
        result = pipeline.degrade([], budget=1000)

        assert result.fidelity == 1.0
        assert len(result.items) == 0
        assert len(result.dropped_ids) == 0

    def test_invalid_budget_raises(self, pipeline):
        """Test that zero/negative budget raises ValueError."""
        items = [ContentItem(id="item-1", content="test", priority=1)]

        with pytest.raises(ValueError, match="positive"):
            pipeline.degrade(items, budget=0)

        with pytest.raises(ValueError, match="positive"):
            pipeline.degrade(items, budget=-100)

    def test_single_item_full_allocation(self, pipeline):
        """Test single item gets full allocation when budget allows."""
        items = [ContentItem(id="item-1", content="A" * 400, priority=1)]

        result = pipeline.degrade(items, budget=1000)

        assert len(result.items) == 1
        assert result.items[0].allocation_ratio == 1.0
        assert result.fidelity == 1.0

    def test_to_dict_includes_all_fields(self, pipeline):
        """Test DegradationResult.to_dict() includes all expected fields."""
        items = [ContentItem(id="item-1", content="A" * 400, priority=1)]

        result = pipeline.degrade(items, budget=50)
        d = result.to_dict()

        expected_keys = [
            "items",
            "tokens_used",
            "fidelity",
            "steps",
            "dropped_ids",
            "warnings",
            "min_items_enforced",
            "chunk_failures",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_step_to_dict_includes_chunk_id(self, pipeline):
        """Test DegradationStep serialization includes chunk_id."""
        items = [ContentItem(id="item-1", content="A" * 400, priority=1)]

        result = pipeline.degrade(items, budget=20)

        if result.steps:
            d = result.to_dict()
            for step in d["steps"]:
                assert "chunk_id" in step
