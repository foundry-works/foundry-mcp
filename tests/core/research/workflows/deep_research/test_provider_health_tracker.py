"""Tests for ProviderHealthTracker in topic_research.py.

Covers:
- Recording successes and failures
- Degraded threshold detection
- all_degraded detection
- Summary format for audit/confidence
"""

from __future__ import annotations

from foundry_mcp.core.research.workflows.deep_research.phases.topic_research import (
    ProviderHealthTracker,
)


class TestProviderHealthTrackerRecordsSuccessAndFailure:
    """Verify basic recording and failure rate calculation."""

    def test_records_success(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("tavily")

        assert tracker.failure_rate("tavily") == 0.0

    def test_records_failure(self):
        tracker = ProviderHealthTracker()
        tracker.record_failure("tavily", "timeout")

        assert tracker.failure_rate("tavily") == 1.0

    def test_mixed_success_and_failure(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("tavily")
        tracker.record_failure("tavily", "search_provider_error")
        tracker.record_success("tavily")
        tracker.record_failure("tavily", "timeout")

        assert tracker.failure_rate("tavily") == 0.5

    def test_unknown_provider_rate_is_zero(self):
        tracker = ProviderHealthTracker()

        assert tracker.failure_rate("unknown") == 0.0

    def test_multiple_providers_tracked_independently(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("tavily")
        tracker.record_failure("perplexity", "timeout")

        assert tracker.failure_rate("tavily") == 0.0
        assert tracker.failure_rate("perplexity") == 1.0


class TestProviderHealthTrackerDegradedThreshold:
    """Verify is_degraded uses the threshold correctly."""

    def test_not_degraded_below_threshold(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        # 7 successes, 3 failures = 30% failure rate
        for _ in range(7):
            tracker.record_success("tavily")
        for _ in range(3):
            tracker.record_failure("tavily", "error")

        assert not tracker.is_degraded("tavily")

    def test_degraded_at_threshold(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        # 2 successes, 8 failures = 80% failure rate
        for _ in range(2):
            tracker.record_success("tavily")
        for _ in range(8):
            tracker.record_failure("tavily", "error")

        assert tracker.is_degraded("tavily")

    def test_degraded_above_threshold(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        # All failures
        for _ in range(5):
            tracker.record_failure("tavily", "error")

        assert tracker.is_degraded("tavily")

    def test_not_degraded_when_no_calls(self):
        tracker = ProviderHealthTracker()

        assert not tracker.is_degraded("tavily")

    def test_window_size_limits_recent_history(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8, window_size=5)
        # First add 10 failures (old history)
        for _ in range(10):
            tracker.record_failure("tavily", "error")
        # Then add 5 successes (fills the window)
        for _ in range(5):
            tracker.record_success("tavily")

        # Window only sees the 5 recent successes
        assert tracker.failure_rate("tavily") == 0.0
        assert not tracker.is_degraded("tavily")

    def test_custom_threshold(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.5)
        # 1 success, 1 failure = 50% failure rate
        tracker.record_success("tavily")
        tracker.record_failure("tavily", "error")

        assert tracker.is_degraded("tavily")


class TestProviderHealthTrackerAllDegraded:
    """Verify all_degraded checks all tracked providers."""

    def test_all_degraded_when_all_fail(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        for _ in range(5):
            tracker.record_failure("tavily", "error")
            tracker.record_failure("perplexity", "timeout")

        assert tracker.all_degraded()

    def test_not_all_degraded_when_one_healthy(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        for _ in range(5):
            tracker.record_failure("tavily", "error")
            tracker.record_success("perplexity")

        assert not tracker.all_degraded()

    def test_not_all_degraded_when_no_providers(self):
        tracker = ProviderHealthTracker()

        assert not tracker.all_degraded()

    def test_single_provider_degraded_means_all_degraded(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        for _ in range(5):
            tracker.record_failure("tavily", "error")

        assert tracker.all_degraded()


class TestProviderHealthTrackerSummary:
    """Verify summary format for audit/confidence section."""

    def test_summary_structure(self):
        tracker = ProviderHealthTracker()
        tracker.record_success("tavily")
        tracker.record_failure("tavily", "timeout")
        tracker.record_success("perplexity")

        summary = tracker.summary()

        assert "providers" in summary
        assert "all_degraded" in summary
        assert summary["all_degraded"] is False

        tavily = summary["providers"]["tavily"]
        assert tavily["total_calls"] == 2
        assert tavily["recent_failures"] == 1
        assert tavily["failure_rate"] == 0.5
        assert tavily["degraded"] is False
        assert tavily["error_types"] == {"timeout": 1}

        perplexity = summary["providers"]["perplexity"]
        assert perplexity["total_calls"] == 1
        assert perplexity["recent_failures"] == 0
        assert perplexity["failure_rate"] == 0.0
        assert perplexity["degraded"] is False

    def test_summary_empty_tracker(self):
        tracker = ProviderHealthTracker()

        summary = tracker.summary()

        assert summary == {"providers": {}, "all_degraded": False}

    def test_summary_error_types_aggregated(self):
        tracker = ProviderHealthTracker()
        tracker.record_failure("tavily", "timeout")
        tracker.record_failure("tavily", "timeout")
        tracker.record_failure("tavily", "search_provider_error")

        summary = tracker.summary()

        error_types = summary["providers"]["tavily"]["error_types"]
        assert error_types == {"timeout": 2, "search_provider_error": 1}

    def test_summary_all_degraded_reflected(self):
        tracker = ProviderHealthTracker(degraded_threshold=0.8)
        for _ in range(5):
            tracker.record_failure("tavily", "error")

        summary = tracker.summary()

        assert summary["all_degraded"] is True
        assert summary["providers"]["tavily"]["degraded"] is True
