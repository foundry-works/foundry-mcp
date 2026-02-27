"""Tests for Phase 3: Token Budget & Truncation Robustness.

Covers:
- 3a: Guard against zero/negative token budgets in truncation
- 3b: Account for token_safety_margin in supplementary notes injection
- 3c: Fix model token limit substring matching order (defense-in-depth)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
    estimate_token_limit_for_model,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    CHARS_PER_TOKEN,
    truncate_to_token_estimate,
)


# =============================================================================
# 3a: Guard against zero/negative token budgets
# =============================================================================


class TestTruncateToTokenEstimateZeroBudget:
    """truncate_to_token_estimate returns full text on zero/negative budgets."""

    def test_zero_max_tokens_returns_full_text(self, caplog: pytest.LogCaptureFixture) -> None:
        text = "Some meaningful content that should not be erased."
        with caplog.at_level(logging.WARNING):
            result = truncate_to_token_estimate(text, 0)
        assert result == text
        assert "max_tokens=0" in caplog.text
        assert "returning full text" in caplog.text

    def test_negative_max_tokens_returns_full_text(self, caplog: pytest.LogCaptureFixture) -> None:
        text = "Another piece of content."
        with caplog.at_level(logging.WARNING):
            result = truncate_to_token_estimate(text, -5)
        assert result == text
        assert "max_tokens=-5" in caplog.text

    def test_positive_budget_still_truncates(self) -> None:
        """Positive budgets still truncate as before."""
        text = "x" * 1000
        result = truncate_to_token_estimate(text, 10)  # 10 tokens * 4 = 40 chars
        assert len(result) < len(text)

    def test_positive_budget_short_text_unchanged(self) -> None:
        """Short text within budget is returned unchanged."""
        text = "short"
        result = truncate_to_token_estimate(text, 1000)
        assert result == text


# =============================================================================
# 3b: Account for token_safety_margin in supplementary notes injection
# =============================================================================


class TestSupplementaryNotesTokenSafetyMargin:
    """_inject_supplementary_raw_notes respects token_safety_margin."""

    def _make_mixin(self, *, token_safety_margin: float = 0.15):
        """Create a minimal SynthesisPhaseMixin-like object with config."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        mixin = object.__new__(SynthesisPhaseMixin)
        config = MagicMock()
        config.token_safety_margin = token_safety_margin
        mixin.config = config
        return mixin

    def _make_state(self, *, raw_notes: list[str] | None = None, synthesis_model: str = "test-model"):
        from foundry_mcp.core.research.models.deep_research import DeepResearchState
        state = DeepResearchState(
            id="test-supplementary",
            original_query="test query",
        )
        state.synthesis_model = synthesis_model
        if raw_notes:
            state.raw_notes = raw_notes
        return state

    @patch(
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.get_model_token_limits",
        return_value={},
    )
    @patch(
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.estimate_token_limit_for_model",
    )
    def test_margin_reduces_effective_context(
        self, mock_estimate, mock_limits,
    ) -> None:
        """With safety margin, headroom is computed from effective context,
        not raw context. A prompt that fits under raw context but not under
        effective context should skip injection."""
        from foundry_mcp.core.research.workflows.deep_research._constants import (
            SYNTHESIS_OUTPUT_RESERVED,
        )

        # Raw context = 100,000 tokens
        # Safety margin = 15% → effective = 85,000
        # SYNTHESIS_OUTPUT_RESERVED = 8,000
        # If prompt takes 76,000 tokens worth of chars:
        #   Raw headroom = 100000 - 76000 - 8000 = 16000 (16% of 100k — would inject)
        #   Effective headroom = 85000 - 76000 - 8000 = 1000 (1.2% of 85k — skip)
        mock_estimate.return_value = 100_000

        prompt_tokens = 76_000
        prompt = "x" * (prompt_tokens * CHARS_PER_TOKEN)

        mixin = self._make_mixin(token_safety_margin=0.15)
        state = self._make_state(raw_notes=["Note 1", "Note 2"])

        result = mixin._inject_supplementary_raw_notes(state, prompt)
        # Should return original prompt (injection skipped due to low headroom)
        assert "Supplementary Research Notes" not in result

    @patch(
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.get_model_token_limits",
        return_value={},
    )
    @patch(
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.estimate_token_limit_for_model",
    )
    def test_zero_margin_uses_full_context(
        self, mock_estimate, mock_limits,
    ) -> None:
        """With zero safety margin, effective context equals raw context."""
        mock_estimate.return_value = 100_000

        # Small prompt leaves plenty of headroom
        prompt = "x" * 1000

        mixin = self._make_mixin(token_safety_margin=0.0)
        state = self._make_state(raw_notes=["Important note"])

        result = mixin._inject_supplementary_raw_notes(state, prompt)
        assert "Supplementary Research Notes" in result

    @patch(
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.get_model_token_limits",
        return_value={},
    )
    @patch(
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis.estimate_token_limit_for_model",
    )
    def test_no_raw_notes_returns_prompt(
        self, mock_estimate, mock_limits,
    ) -> None:
        """No raw notes → prompt returned unchanged regardless of margin."""
        mock_estimate.return_value = 100_000
        mixin = self._make_mixin()
        state = self._make_state(raw_notes=None)
        prompt = "original prompt"
        assert mixin._inject_supplementary_raw_notes(state, prompt) == prompt


# =============================================================================
# 3c: Model token limit substring matching — defense-in-depth
# =============================================================================


class TestEstimateTokenLimitLongestMatch:
    """estimate_token_limit_for_model picks the longest matching key."""

    def test_longest_match_wins_unsorted_dict(self) -> None:
        """Even with an unsorted dict, the most specific key wins."""
        # Dict deliberately has shorter key first
        limits = {
            "gpt-4": 8192,
            "gpt-4o": 128_000,
        }
        result = estimate_token_limit_for_model("gpt-4o-2024-08-06", limits)
        assert result == 128_000

    def test_exact_prefix_match(self) -> None:
        limits = {
            "claude-3": 100_000,
            "claude-3.5-sonnet": 200_000,
        }
        result = estimate_token_limit_for_model("claude-3.5-sonnet-20240620", limits)
        assert result == 200_000

    def test_no_match_returns_none(self) -> None:
        limits = {"gpt-4": 8192}
        assert estimate_token_limit_for_model("llama-70b", limits) is None

    def test_none_model_returns_none(self) -> None:
        assert estimate_token_limit_for_model(None, {"gpt-4": 8192}) is None

    def test_empty_model_returns_none(self) -> None:
        assert estimate_token_limit_for_model("", {"gpt-4": 8192}) is None

    def test_case_insensitive_matching(self) -> None:
        limits = {"GPT-4o": 128_000}
        result = estimate_token_limit_for_model("gpt-4o-mini", limits)
        assert result == 128_000

    def test_multiple_matches_picks_longest(self) -> None:
        """When multiple keys match, the longest (most specific) wins."""
        limits = {
            "gpt-4": 8_192,
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
        }
        # "gpt-4o-mini-2024" matches all three; longest key "gpt-4o-mini" wins
        result = estimate_token_limit_for_model("gpt-4o-mini-2024", limits)
        assert result == 128_000
