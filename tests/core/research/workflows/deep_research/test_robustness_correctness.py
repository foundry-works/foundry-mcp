"""Tests for robustness and correctness fixes.

Covers:
- advance_phase() handles consecutive deprecated phases
- deep-research-evaluate in authorization allowlists
- _critique_has_issues hardened verdict/issue parsing
- Researcher conversation history truncation
- Model token limits sorted by longest match
"""

from __future__ import annotations

import pytest

from foundry_mcp.core.authorization import (
    AUTONOMY_RUNNER_ALLOWLIST,
    OBSERVER_ALLOWLIST,
    check_action_allowed,
)
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    _sort_limits_longest_first,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision import (
    SupervisionPhaseMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
    _truncate_researcher_history,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    phase: DeepResearchPhase = DeepResearchPhase.BRIEF,
) -> DeepResearchState:
    """Create a minimal DeepResearchState for phase testing."""
    return DeepResearchState(
        id="deepres-test-phase4",
        original_query="test query",
        phase=phase,
        iteration=1,
        max_iterations=3,
    )


# =============================================================================
# 4A: advance_phase() skips consecutive deprecated phases
# =============================================================================


class TestAdvancePhaseConsecutiveSkip:
    """Test that advance_phase handles consecutive deprecated phases."""

    def test_single_skip_gathering(self) -> None:
        """BRIEF -> SUPERVISION (skipping GATHERING) still works."""
        state = _make_state(phase=DeepResearchPhase.BRIEF)
        result = state.advance_phase()
        assert result == DeepResearchPhase.SUPERVISION
        assert state.phase == DeepResearchPhase.SUPERVISION

    def test_consecutive_deprecated_phases_all_skipped(self) -> None:
        """If multiple consecutive phases are deprecated, all are skipped.

        We temporarily add SUPERVISION to _SKIP_PHASES to simulate
        consecutive deprecated phases (PLANNING + GATHERING + SUPERVISION).
        BRIEF should advance directly to SYNTHESIS.
        """
        state = _make_state(phase=DeepResearchPhase.BRIEF)
        original_skip = DeepResearchState._SKIP_PHASES
        try:
            # Simulate three consecutive deprecated phases
            DeepResearchState._SKIP_PHASES = {
                DeepResearchPhase.PLANNING,
                DeepResearchPhase.GATHERING,
                DeepResearchPhase.SUPERVISION,
            }
            result = state.advance_phase()
            assert result == DeepResearchPhase.SYNTHESIS
            assert state.phase == DeepResearchPhase.SYNTHESIS
        finally:
            DeepResearchState._SKIP_PHASES = original_skip

    def test_advance_at_terminal_phase_stays(self) -> None:
        """Advancing from SYNTHESIS (terminal) doesn't change phase."""
        state = _make_state(phase=DeepResearchPhase.SYNTHESIS)
        result = state.advance_phase()
        assert result == DeepResearchPhase.SYNTHESIS

    def test_normal_advance_clarification_to_brief(self) -> None:
        """Normal advance without any skipping."""
        state = _make_state(phase=DeepResearchPhase.CLARIFICATION)
        result = state.advance_phase()
        assert result == DeepResearchPhase.BRIEF


# =============================================================================
# 4B: Authorization allowlists include deep-research-evaluate
# =============================================================================


class TestDeepResearchEvaluateAuthorization:
    """Test that deep-research-evaluate is in the correct allowlists."""

    def test_observer_allowlist_includes_evaluate(self) -> None:
        """deep-research-evaluate should be in OBSERVER_ALLOWLIST."""
        assert "deep-research-evaluate" in OBSERVER_ALLOWLIST

    def test_autonomy_runner_allowlist_includes_evaluate(self) -> None:
        """deep-research-evaluate should be in AUTONOMY_RUNNER_ALLOWLIST."""
        assert "deep-research-evaluate" in AUTONOMY_RUNNER_ALLOWLIST

    def test_observer_can_evaluate(self) -> None:
        """Observer role can execute deep-research-evaluate."""
        result = check_action_allowed("observer", None, "deep-research-evaluate")
        assert result.allowed is True

    def test_autonomy_runner_can_evaluate(self) -> None:
        """Autonomy runner role can execute deep-research-evaluate."""
        result = check_action_allowed("autonomy_runner", None, "deep-research-evaluate")
        assert result.allowed is True


# =============================================================================
# 4C: _critique_has_issues hardened parsing
# =============================================================================


class TestCritiqueHasIssuesHardened:
    """Test that _critique_has_issues handles formatting variations."""

    def test_verdict_no_issues_standard(self) -> None:
        """Standard VERDICT: NO_ISSUES."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "All good.\nVERDICT: NO_ISSUES"
        ) is False

    def test_verdict_no_issues_no_space(self) -> None:
        """VERDICT:NO_ISSUES without space."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "All criteria pass.\nVERDICT:NO_ISSUES"
        ) is False

    def test_verdict_no_issues_extra_spaces(self) -> None:
        """VERDICT with extra whitespace."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "All fine.\nVERDICT :  NO_ISSUES"
        ) is False

    def test_verdict_no_issues_space_instead_of_underscore(self) -> None:
        """VERDICT: NO ISSUES (space instead of underscore)."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "All fine.\nVERDICT: NO ISSUES"
        ) is False

    def test_verdict_revision_needed_standard(self) -> None:
        """Standard VERDICT: REVISION_NEEDED."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "Problems found.\nVERDICT: REVISION_NEEDED"
        ) is True

    def test_verdict_revision_needed_extra_spaces(self) -> None:
        """VERDICT with extra whitespace before REVISION_NEEDED."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "Problems found.\nVERDICT :  REVISION_NEEDED"
        ) is True

    def test_verdict_case_insensitive(self) -> None:
        """Verdict matching is case insensitive."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "Verdict: no_issues"
        ) is False
        assert SupervisionPhaseMixin._critique_has_issues(
            "verdict: revision_needed"
        ) is True

    def test_issue_marker_at_line_start(self) -> None:
        """ISSUE: at line start is detected."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "ISSUE: directive overlap detected"
        ) is True

    def test_issue_marker_after_label(self) -> None:
        """ISSUE: after a label prefix (e.g. 'Redundancy: ISSUE:') is detected."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "1. Redundancy: ISSUE: directives 1 and 3 overlap"
        ) is True

    def test_conversational_issue_no_colon(self) -> None:
        """Casual mention of 'issue' without colon is NOT detected."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "There is no issue with this approach"
        ) is False

    def test_no_verdict_no_issues_clean(self) -> None:
        """Clean text with no markers returns False."""
        assert SupervisionPhaseMixin._critique_has_issues(
            "Everything looks fine. All criteria pass."
        ) is False


# =============================================================================
# 4E: Researcher conversation history truncation
# =============================================================================


class TestResearcherHistoryTruncation:
    """Test _truncate_researcher_history function."""

    def test_short_history_unchanged(self) -> None:
        """Short history within budget is returned unchanged."""
        history = [
            {"role": "assistant", "content": "thinking..."},
            {"role": "tool", "content": "search results"},
        ]
        result = _truncate_researcher_history(history, "claude-sonnet-4-20250514")
        assert result == history

    def test_below_min_preserve_threshold(self) -> None:
        """History with fewer than MIN_PRESERVE_RECENT_TURNS is returned as-is."""
        history = [
            {"role": "assistant", "content": "x" * 100000},
            {"role": "tool", "content": "y" * 100000},
        ]
        result = _truncate_researcher_history(history, "some-small-model")
        # Only 2 items, below _MIN_PRESERVE_RECENT_TURNS (4), so returned as-is
        assert len(result) == 2

    def test_large_history_truncated(self) -> None:
        """History exceeding context budget is truncated from the front."""
        # Create a large history that will exceed any reasonable model budget
        # Use a tiny model limit to force truncation
        history = [
            {"role": "assistant", "content": f"Turn {i}: " + "x" * 5000}
            for i in range(20)
        ]
        # Use None model to trigger 128K fallback
        # Budget = 128000 * 0.35 * 4 = 179,200 chars
        # 20 * 5000 = 100,000 chars — fits within budget
        # So let's make messages bigger
        history = [
            {"role": "assistant", "content": f"Turn {i}: " + "x" * 50000}
            for i in range(20)
        ]
        # 20 * 50000 = 1,000,000 chars — exceeds 179,200 char budget
        result = _truncate_researcher_history(history, None)
        assert len(result) < len(history)
        # Most recent turns are preserved
        assert result[-1] == history[-1]

    def test_preserves_recent_turns(self) -> None:
        """Truncation preserves at least _MIN_PRESERVE_RECENT_TURNS entries."""
        from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
            _MIN_PRESERVE_RECENT_TURNS,
        )
        history = [
            {"role": "assistant", "content": "x" * 100000}
            for _ in range(10)
        ]
        result = _truncate_researcher_history(history, None)
        assert len(result) >= _MIN_PRESERVE_RECENT_TURNS


# =============================================================================
# 4F: Model token limits sorted by longest match
# =============================================================================


class TestTokenLimitsSortedByLongestMatch:
    """Test that token limits are sorted by key length descending."""

    def test_sort_limits_longest_first(self) -> None:
        """_sort_limits_longest_first puts longer keys first."""
        limits = {
            "gpt-4": 8192,
            "gpt-4.1-mini": 1048576,
            "gpt-4.1": 1048576,
        }
        sorted_limits = _sort_limits_longest_first(limits)
        keys = list(sorted_limits.keys())
        assert keys[0] == "gpt-4.1-mini"  # longest (13 chars)
        assert keys[1] == "gpt-4.1"       # 7 chars
        assert keys[2] == "gpt-4"         # 5 chars

    def test_loaded_limits_are_sorted(self) -> None:
        """MODEL_TOKEN_LIMITS loaded at import time are sorted longest-first."""
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            MODEL_TOKEN_LIMITS,
        )
        keys = list(MODEL_TOKEN_LIMITS.keys())
        if len(keys) > 1:
            # Verify that keys are sorted by length descending
            for i in range(len(keys) - 1):
                assert len(keys[i]) >= len(keys[i + 1]), (
                    f"Key {keys[i]!r} (len {len(keys[i])}) should come before "
                    f"{keys[i + 1]!r} (len {len(keys[i + 1])})"
                )

    def test_longest_match_wins(self) -> None:
        """With sorted limits, more specific patterns match first."""
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            estimate_token_limit_for_model,
        )
        # gpt-4.1-mini should match its specific entry, not the shorter gpt-4.1
        limits = _sort_limits_longest_first({
            "gpt-4.1": 1048576,
            "gpt-4.1-mini": 200000,
            "gpt-4": 8192,
        })
        result = estimate_token_limit_for_model("gpt-4.1-mini-2025-04-14", limits)
        assert result == 200000  # specific match, not the shorter gpt-4.1
