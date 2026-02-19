"""Integration tests for deep research token management.

Tests verify:
1. Graceful degradation with artificially low model limits
2. Allocation across analysis → synthesis → refinement phases
3. Minimum item guardrails (min 3 items per phase when possible)
4. Fidelity metadata accuracy in responses
"""

from datetime import datetime, timezone

import pytest

from foundry_mcp.core.research.context_budget import (
    AllocationStrategy,
    ContentItem,
    ContextBudgetManager,
)
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.sources import ResearchSource, SourceQuality
from foundry_mcp.core.research.token_management import (
    BudgetingMode,
    ModelContextLimits,
    TokenBudget,
    get_effective_context,
    preflight_count,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_model_limits() -> ModelContextLimits:
    """Create artificially small model limits for testing degradation."""
    return ModelContextLimits(
        context_window=2000,  # Very small: 2K tokens
        max_output_tokens=500,
        budgeting_mode=BudgetingMode.INPUT_ONLY,
    )


@pytest.fixture
def small_model_limits() -> ModelContextLimits:
    """Create small but usable model limits."""
    return ModelContextLimits(
        context_window=10_000,  # 10K tokens
        max_output_tokens=2000,
        budgeting_mode=BudgetingMode.INPUT_ONLY,
    )


@pytest.fixture
def mock_sources() -> list[ResearchSource]:
    """Create mock research sources with varying sizes and quality."""
    now = datetime.now(timezone.utc)
    return [
        ResearchSource(
            id="src-high-1",
            url="https://example.com/article1",
            title="High Quality Source 1",
            snippet="Brief snippet",
            content="A" * 2000,  # ~500 tokens
            quality=SourceQuality.HIGH,
            discovered_at=now,
        ),
        ResearchSource(
            id="src-high-2",
            url="https://example.com/article2",
            title="High Quality Source 2",
            snippet="Brief snippet",
            content="B" * 1600,  # ~400 tokens
            quality=SourceQuality.HIGH,
            discovered_at=now,
        ),
        ResearchSource(
            id="src-med-1",
            url="https://example.com/article3",
            title="Medium Quality Source 1",
            snippet="Brief snippet",
            content="C" * 1200,  # ~300 tokens
            quality=SourceQuality.MEDIUM,
            discovered_at=now,
        ),
        ResearchSource(
            id="src-med-2",
            url="https://example.com/article4",
            title="Medium Quality Source 2",
            snippet="Brief snippet",
            content="D" * 800,  # ~200 tokens
            quality=SourceQuality.MEDIUM,
            discovered_at=now,
        ),
        ResearchSource(
            id="src-low-1",
            url="https://example.com/article5",
            title="Low Quality Source 1",
            snippet="Brief snippet",
            content="E" * 600,  # ~150 tokens
            quality=SourceQuality.LOW,
            discovered_at=now,
        ),
    ]


@pytest.fixture
def mock_research_state(mock_sources) -> DeepResearchState:
    """Create a mock deep research state with sources."""
    return DeepResearchState(
        id="test-research-001",
        original_query="Test research query",
        sources=mock_sources,
        phase=DeepResearchPhase.ANALYSIS,
        analysis_provider="claude",
        synthesis_provider="claude",
    )


@pytest.fixture
def fixed_token_manager() -> ContextBudgetManager:
    """Create a manager with fixed 4 chars/token estimation."""
    return ContextBudgetManager(token_estimator=lambda content: max(1, len(content) // 4))


# =============================================================================
# Test: Graceful Degradation with Low Limits
# =============================================================================


class TestGracefulDegradationWithLowLimits:
    """Tests for graceful degradation under artificially low model limits."""

    def test_degradation_drops_low_priority_first(self, fixed_token_manager, mock_sources):
        """Test that low-priority sources are dropped first under tight budget."""
        # Convert sources to content items (mimicking _allocate_source_budget)
        items = []
        for i, source in enumerate(mock_sources):
            # Priority: HIGH=1, MEDIUM=2, LOW=3
            priority_map = {
                SourceQuality.HIGH: 1,
                SourceQuality.MEDIUM: 2,
                SourceQuality.LOW: 3,
            }
            priority = priority_map.get(source.quality, 3)
            items.append(
                ContentItem(
                    id=source.id,
                    content=source.content or source.snippet or "",
                    priority=priority,
                    source_id=source.id,
                    protected=source.quality == SourceQuality.HIGH,
                )
            )

        # Total: ~1550 tokens (500+400+300+200+150)
        # Budget: 1000 tokens - should drop low priority items
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=1000,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # High quality sources should be preserved
        allocated_ids = {item.id for item in result.items}
        assert "src-high-1" in allocated_ids
        assert "src-high-2" in allocated_ids

        # Low quality source should be dropped
        assert "src-low-1" in result.dropped_ids or "src-low-1" not in allocated_ids

        # Fidelity should be less than 1.0 due to drops/compression
        assert result.fidelity < 1.0

    def test_degradation_with_very_tight_budget(self, fixed_token_manager):
        """Test behavior with extremely tight budget (less than smallest item)."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="y" * 400, priority=2),  # 100 tokens
            ContentItem(id="c", content="z" * 400, priority=3),  # 100 tokens
        ]

        # Budget of only 50 tokens - can't fit any item fully
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=50,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Should still allocate first item with partial budget
        assert len(result.items) >= 1
        first_item = result.items[0]
        assert first_item.id == "a"
        assert first_item.needs_summarization

        # Fidelity should be very low
        assert result.fidelity < 0.5

    def test_protected_items_preserved_under_tight_budget(self, fixed_token_manager):
        """Test that protected items are never dropped, only compressed."""
        items = [
            ContentItem(id="big", content="A" * 4000, priority=1),  # 1000 tokens
            ContentItem(id="protected", content="B" * 400, priority=2, protected=True),
            ContentItem(id="regular", content="C" * 400, priority=3),
        ]

        # Budget exhausted by first item
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=1000,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Protected item must be allocated, not dropped
        protected = next((i for i in result.items if i.id == "protected"), None)
        assert protected is not None
        assert "protected" not in result.dropped_ids

        # Regular low-priority item may be dropped
        assert "regular" in result.dropped_ids


class TestMinimumItemGuardrails:
    """Tests for minimum item guardrails (aim for min 3 items when possible)."""

    def test_equal_share_preserves_all_items(self, fixed_token_manager):
        """Test EQUAL_SHARE strategy preserves all items even with compression."""
        items = [
            ContentItem(id="a", content="A" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="B" * 600, priority=2),  # 150 tokens
            ContentItem(id="c", content="C" * 800, priority=3),  # 200 tokens
            ContentItem(id="d", content="D" * 400, priority=4),  # 100 tokens
        ]

        # Total: 550 tokens, Budget: 300 tokens
        # Equal share should not drop any items, just compress
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=300,
            strategy=AllocationStrategy.EQUAL_SHARE,
        )

        # All items should be preserved (with compression)
        assert len(result.items) == 4
        assert len(result.dropped_ids) == 0

    def test_proportional_preserves_all_items(self, fixed_token_manager):
        """Test PROPORTIONAL strategy preserves all items with compression."""
        items = [
            ContentItem(id="a", content="A" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="B" * 400, priority=2),  # 100 tokens
            ContentItem(id="c", content="C" * 400, priority=3),  # 100 tokens
        ]

        # Total: 300 tokens, Budget: 150 tokens (50% compression)
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=150,
            strategy=AllocationStrategy.PROPORTIONAL,
        )

        # All items should be preserved
        assert len(result.items) == 3
        assert len(result.dropped_ids) == 0

        # Each item should be at ~50% allocation
        for item in result.items:
            assert 0.45 <= item.allocation_ratio <= 0.55

    def test_priority_first_with_many_small_items_preserves_more(self, fixed_token_manager):
        """Test that having many small items allows more to be preserved."""
        # 6 small items instead of 3 large ones
        items = [ContentItem(id=f"item-{i}", content="X" * 200, priority=i + 1) for i in range(6)]

        # Each item is 50 tokens, total 300 tokens
        # Budget of 200 should fit 4 items fully
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=200,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Should preserve at least 4 items (200 / 50 = 4)
        assert len(result.items) >= 4


# =============================================================================
# Test: Phase Budget Calculations
# =============================================================================


class TestPhaseBudgetCalculations:
    """Tests for phase-specific budget calculations."""

    def test_analysis_phase_budget_fraction(self, small_model_limits):
        """Test analysis phase uses 80% of effective context."""
        from foundry_mcp.core.research.workflows.deep_research import (
            ANALYSIS_OUTPUT_RESERVED,
            ANALYSIS_PHASE_BUDGET_FRACTION,
        )

        effective = get_effective_context(small_model_limits, output_budget=ANALYSIS_OUTPUT_RESERVED)
        phase_budget = int(effective * ANALYSIS_PHASE_BUDGET_FRACTION)

        # Should be 80% of 10K = 8K tokens
        assert phase_budget == int(10_000 * 0.80)

    def test_synthesis_phase_budget_fraction(self, small_model_limits):
        """Test synthesis phase uses 85% of effective context."""
        from foundry_mcp.core.research.workflows.deep_research import (
            SYNTHESIS_OUTPUT_RESERVED,
            SYNTHESIS_PHASE_BUDGET_FRACTION,
        )

        effective = get_effective_context(small_model_limits, output_budget=SYNTHESIS_OUTPUT_RESERVED)
        phase_budget = int(effective * SYNTHESIS_PHASE_BUDGET_FRACTION)

        # Should be 85% of 10K = 8.5K tokens
        assert phase_budget == int(10_000 * 0.85)

    def test_refinement_phase_budget_fraction(self, small_model_limits):
        """Test refinement phase uses 70% of effective context."""
        from foundry_mcp.core.research.workflows.deep_research import (
            REFINEMENT_OUTPUT_RESERVED,
            REFINEMENT_PHASE_BUDGET_FRACTION,
        )

        effective = get_effective_context(small_model_limits, output_budget=REFINEMENT_OUTPUT_RESERVED)
        phase_budget = int(effective * REFINEMENT_PHASE_BUDGET_FRACTION)

        # Should be 70% of 10K = 7K tokens
        assert phase_budget == int(10_000 * 0.70)

    def test_tiny_limits_still_provide_usable_budget(self, tiny_model_limits):
        """Test that even tiny limits provide some usable budget."""
        from foundry_mcp.core.research.workflows.deep_research import (
            ANALYSIS_OUTPUT_RESERVED,
            ANALYSIS_PHASE_BUDGET_FRACTION,
        )

        effective = get_effective_context(tiny_model_limits, output_budget=ANALYSIS_OUTPUT_RESERVED)
        phase_budget = int(effective * ANALYSIS_PHASE_BUDGET_FRACTION)

        # 2000 * 0.80 = 1600 tokens
        assert phase_budget == int(2000 * 0.80)
        # Should be enough for at least a few small sources
        assert phase_budget >= 1000


# =============================================================================
# Test: Fidelity Metadata Accuracy
# =============================================================================


class TestFidelityMetadataAccuracy:
    """Tests for fidelity metadata accuracy in allocation results."""

    def test_full_fidelity_when_budget_exceeds_content(self, fixed_token_manager):
        """Test fidelity is 1.0 when budget exceeds total content."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="y" * 400, priority=2),  # 100 tokens
        ]

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=1000,  # Way more than needed
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        assert result.fidelity == 1.0
        assert result.tokens_used == 200
        assert len(result.dropped_ids) == 0
        for item in result.items:
            assert item.allocation_ratio == 1.0
            assert not item.needs_summarization

    def test_partial_fidelity_with_drops(self, fixed_token_manager):
        """Test fidelity reflects dropped content."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="y" * 400, priority=2),  # 100 tokens
            ContentItem(id="c", content="z" * 400, priority=3),  # 100 tokens
        ]

        # Budget for only 2 items
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=200,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Fidelity should be 2/3 = 0.666...
        assert 0.65 <= result.fidelity <= 0.68
        assert len(result.dropped_ids) == 1
        assert "c" in result.dropped_ids

    def test_partial_fidelity_with_compression(self, fixed_token_manager):
        """Test fidelity reflects compressed content."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
        ]

        # Budget for only half
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=50,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Fidelity should be 0.5
        assert result.fidelity == 0.5
        assert result.items[0].allocation_ratio == 0.5
        assert result.items[0].needs_summarization

    def test_fidelity_level_conversion(self):
        """Test fidelity score to level string conversion."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            fidelity_level_from_score,
        )

        # Test thresholds
        assert fidelity_level_from_score(1.0) == "full"
        assert fidelity_level_from_score(0.95) == "full"
        assert fidelity_level_from_score(0.9) == "full"
        assert fidelity_level_from_score(0.89) == "condensed"
        assert fidelity_level_from_score(0.6) == "condensed"
        assert fidelity_level_from_score(0.59) == "compressed"
        assert fidelity_level_from_score(0.3) == "compressed"
        assert fidelity_level_from_score(0.29) == "minimal"
        assert fidelity_level_from_score(0.0) == "minimal"

    def test_to_dict_includes_all_fidelity_fields(self, fixed_token_manager):
        """Test AllocationResult.to_dict includes all fidelity metadata."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),
            ContentItem(id="b", content="y" * 400, priority=2),
        ]

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=150,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        d = result.to_dict()

        # Check all required fidelity fields
        assert "fidelity" in d
        assert "tokens_used" in d
        assert "tokens_available" in d
        assert "utilization" in d
        assert "dropped_ids" in d
        assert "items_allocated" in d
        assert "items_dropped" in d
        assert "items" in d

        # Check item-level fidelity fields
        for item in d["items"]:
            assert "allocation_ratio" in item
            assert "needs_summarization" in item
            assert "original_tokens" in item
            assert "allocated_tokens" in item


# =============================================================================
# Test: Cross-Phase Degradation
# =============================================================================


class TestCrossPhaseGracefulDegradation:
    """Tests for graceful degradation across multiple phases."""

    @pytest.fixture
    def phase_budgets(self, tiny_model_limits):
        """Calculate phase budgets for tiny model limits."""
        from foundry_mcp.core.research.workflows.deep_research import (
            ANALYSIS_PHASE_BUDGET_FRACTION,
            REFINEMENT_PHASE_BUDGET_FRACTION,
            SYNTHESIS_PHASE_BUDGET_FRACTION,
        )

        effective = get_effective_context(tiny_model_limits)
        return {
            "analysis": int(effective * ANALYSIS_PHASE_BUDGET_FRACTION),
            "synthesis": int(effective * SYNTHESIS_PHASE_BUDGET_FRACTION),
            "refinement": int(effective * REFINEMENT_PHASE_BUDGET_FRACTION),
        }

    def test_analysis_phase_handles_budget_pressure(self, fixed_token_manager, phase_budgets):
        """Test analysis phase gracefully handles tight budget."""
        # Simulate sources that exceed analysis budget
        items = [ContentItem(id=f"src-{i}", content="X" * 800, priority=i + 1) for i in range(5)]
        # Total: 1000 tokens (5 * 200)

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=phase_budgets["analysis"],  # 1600 tokens
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # All items should fit (1000 < 1600)
        assert len(result.items) == 5
        assert result.fidelity == 1.0

    def test_synthesis_phase_handles_budget_pressure(self, fixed_token_manager, phase_budgets):
        """Test synthesis phase gracefully handles tight budget."""
        # Simulate findings that exceed synthesis budget
        items = [ContentItem(id=f"finding-{i}", content="X" * 1600, priority=1) for i in range(5)]
        # Total: 2000 tokens (5 * 400)

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=phase_budgets["synthesis"],  # 1700 tokens
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Some items should be dropped or compressed
        assert result.fidelity < 1.0
        # But we should have preserved high-priority content
        assert len(result.items) >= 3

    def test_refinement_phase_most_constrained(self, fixed_token_manager, phase_budgets):
        """Test refinement phase has smallest budget (70%)."""
        assert phase_budgets["refinement"] < phase_budgets["analysis"]
        assert phase_budgets["refinement"] < phase_budgets["synthesis"]

        # With tighter budget, should drop/compress more aggressively
        items = [ContentItem(id=f"item-{i}", content="X" * 800, priority=i + 1) for i in range(5)]

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=phase_budgets["refinement"],  # 1400 tokens
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Even with constraints, should still produce valid output
        assert len(result.items) >= 1
        assert result.tokens_used <= phase_budgets["refinement"]


# =============================================================================
# Test: State Fidelity Tracking
# =============================================================================


class TestStateFidelityTracking:
    """Tests for fidelity tracking in DeepResearchState."""

    def test_state_has_fidelity_fields(self):
        """Test DeepResearchState includes fidelity tracking fields."""
        state = DeepResearchState(
            id="test",
            original_query="test query",
        )

        # Check default values - content_fidelity is now a dict
        assert state.content_fidelity == {}
        assert state.dropped_content_ids == []
        assert state.content_allocation_metadata == {}

    def test_state_fidelity_updates(self):
        """Test fidelity fields can be updated via record_item_fidelity."""
        from foundry_mcp.core.research.models.fidelity import FidelityLevel

        state = DeepResearchState(
            id="test",
            original_query="test query",
        )

        # Use the new record_item_fidelity method
        state.record_item_fidelity(
            item_id="src-1",
            phase="analysis",
            level=FidelityLevel.CONDENSED,
            reason="budget_exceeded",
        )
        state.dropped_content_ids = ["src-2", "src-3"]
        state.content_allocation_metadata = {
            "tokens_used": 5000,
            "fidelity": 0.75,
        }

        # Verify the new structure
        assert "src-1" in state.content_fidelity
        assert state.content_fidelity["src-1"].current_level == FidelityLevel.CONDENSED
        assert len(state.dropped_content_ids) == 2
        assert state.content_allocation_metadata["fidelity"] == 0.75


# =============================================================================
# Test: Budget with Different Model Limits
# =============================================================================


class TestBudgetWithDifferentModelLimits:
    """Tests for budget allocation with different model configurations."""

    def test_budget_scales_with_model_size(self):
        """Test that budget scales appropriately with model context window."""
        from foundry_mcp.core.research.workflows.deep_research import (
            ANALYSIS_PHASE_BUDGET_FRACTION,
        )

        # Small model: 10K context
        small_limits = ModelContextLimits(
            context_window=10_000,
            max_output_tokens=2000,
        )
        small_effective = get_effective_context(small_limits)
        small_budget = int(small_effective * ANALYSIS_PHASE_BUDGET_FRACTION)

        # Large model: 200K context
        large_limits = ModelContextLimits(
            context_window=200_000,
            max_output_tokens=32_000,
        )
        large_effective = get_effective_context(large_limits)
        large_budget = int(large_effective * ANALYSIS_PHASE_BUDGET_FRACTION)

        # Large model should have ~20x the budget
        ratio = large_budget / small_budget
        assert 15 <= ratio <= 25

    def test_combined_budgeting_mode_reserves_output(self):
        """Test COMBINED mode properly reserves output tokens."""
        combined_limits = ModelContextLimits(
            context_window=10_000,
            max_output_tokens=2000,
            budgeting_mode=BudgetingMode.COMBINED,
            output_reserved=2000,
        )

        effective = get_effective_context(combined_limits)

        # Should reserve output from context: 10000 - 2000 = 8000
        assert effective == 8000

    def test_input_only_mode_uses_full_context(self):
        """Test INPUT_ONLY mode uses full context window."""
        input_only_limits = ModelContextLimits(
            context_window=10_000,
            max_output_tokens=2000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        )

        effective = get_effective_context(input_only_limits)

        # Should use full context window
        assert effective == 10_000


# =============================================================================
# Test: Preflight Validation Under Pressure
# =============================================================================


class TestPreflightValidationUnderPressure:
    """Tests for preflight validation with tight budgets."""

    def test_preflight_detects_overflow(self):
        """Test preflight correctly detects when content exceeds budget."""
        budget = TokenBudget(total_budget=1000, safety_margin=0.0)
        content = "x" * 8000  # ~2000 tokens

        result = preflight_count(content, budget, warn_on_heuristic=False)

        assert result.valid is False
        assert result.overflow_tokens > 0
        assert result.estimated_tokens > budget.remaining()

    def test_preflight_with_safety_margin(self):
        """Test preflight respects safety margin."""
        budget = TokenBudget(
            total_budget=1000,
            safety_margin=0.2,  # 20% safety margin
        )
        # Effective budget: 1000 * 0.8 = 800 tokens

        # Content just under effective budget
        content = "x" * 3000  # ~750 tokens

        result = preflight_count(content, budget, warn_on_heuristic=False)

        assert result.valid is True
        assert result.estimated_tokens < budget.effective_budget()

    def test_preflight_final_fit_flag(self):
        """Test preflight is_final_fit flag is preserved."""
        budget = TokenBudget(total_budget=1000, safety_margin=0.0)
        content = "x" * 400  # ~100 tokens

        result = preflight_count(content, budget, is_final_fit=True, warn_on_heuristic=False)

        assert result.is_final_fit is True

    def test_preflight_usage_fraction(self):
        """Test preflight usage_fraction property."""
        budget = TokenBudget(total_budget=1000, safety_margin=0.0)
        content = "x" * 400  # ~100 tokens

        result = preflight_count(content, budget, warn_on_heuristic=False)

        # 100 / 1000 = 0.1
        assert result.usage_fraction == 0.1


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestTokenIntegrationEdgeCases:
    """Tests for edge cases in token integration."""

    def test_empty_sources_handled(self, fixed_token_manager):
        """Test allocation with empty sources list."""
        result = fixed_token_manager.allocate_budget(
            items=[],
            budget=1000,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        assert len(result.items) == 0
        assert result.fidelity == 1.0
        assert result.tokens_used == 0

    def test_single_item_larger_than_budget(self, fixed_token_manager):
        """Test single item larger than entire budget."""
        items = [
            ContentItem(id="huge", content="x" * 10000, priority=1),  # 2500 tokens
        ]

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=500,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # Item should be allocated with compression
        assert len(result.items) == 1
        assert result.items[0].needs_summarization
        assert result.items[0].allocated_tokens == 500
        assert result.fidelity < 0.25  # 500 / 2500 = 0.2

    def test_all_items_protected_under_pressure(self, fixed_token_manager):
        """Test behavior when all items are protected under tight budget."""
        items = [
            ContentItem(id="p1", content="A" * 400, priority=1, protected=True),
            ContentItem(id="p2", content="B" * 400, priority=2, protected=True),
            ContentItem(id="p3", content="C" * 400, priority=3, protected=True),
        ]
        # Total: 300 tokens

        # Budget only allows 1.5 items
        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=150,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # All protected items should be present (not dropped)
        assert len(result.dropped_ids) == 0
        allocated_ids = {i.id for i in result.items}
        assert "p1" in allocated_ids
        assert "p2" in allocated_ids
        assert "p3" in allocated_ids

    def test_mixed_empty_and_content_items(self, fixed_token_manager):
        """Test allocation with mix of empty and content items."""
        items = [
            ContentItem(id="full", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="empty", content="", priority=2),  # 0 tokens
            ContentItem(id="snippet", content="y" * 40, priority=3),  # 10 tokens
        ]

        result = fixed_token_manager.allocate_budget(
            items=items,
            budget=200,
            strategy=AllocationStrategy.PRIORITY_FIRST,
        )

        # All items should fit
        assert len(result.items) == 3
        assert result.tokens_used <= 200
