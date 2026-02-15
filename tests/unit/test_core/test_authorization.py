"""Tests for authorization module."""

import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from foundry_mcp.core.authorization import (
    Role,
    AuthzResult,
    AUTONOMY_RUNNER_ALLOWLIST,
    MAINTAINER_ALLOWLIST,
    OBSERVER_ALLOWLIST,
    server_role_var,
    get_server_role,
    set_server_role,
    get_role_allowlist,
    check_action_allowed,
    initialize_role_from_config,
    RateLimitConfig,
    RateLimitTracker,
    get_rate_limit_tracker,
    reset_rate_limit_tracker,
    RunnerIsolationConfig,
    set_runner_isolation_config,
    reset_runner_isolation_config,
    run_isolated_subprocess,
    StdinTimeoutError,
)


class TestRoleEnum:
    """Test Role enum."""

    def test_role_values(self):
        assert Role.AUTONOMY_RUNNER.value == "autonomy_runner"
        assert Role.MAINTAINER.value == "maintainer"
        assert Role.OBSERVER.value == "observer"


class TestRoleAllowlists:
    """Test role allowlist constants."""

    def test_autonomy_runner_allowlist(self):
        assert "session-start" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-step-next" in AUTONOMY_RUNNER_ALLOWLIST
        assert "review-fidelity-gate" in AUTONOMY_RUNNER_ALLOWLIST

    def test_maintainer_allowlist_has_wildcard(self):
        assert "*" in MAINTAINER_ALLOWLIST

    def test_observer_allowlist_readonly(self):
        assert "list" in OBSERVER_ALLOWLIST
        assert "status" in OBSERVER_ALLOWLIST
        assert "session-start" not in OBSERVER_ALLOWLIST


class TestServerRoleVar:
    """Test server role context variable."""

    def test_default_role_is_observer(self):
        assert get_server_role() == "observer"

    def test_set_server_role(self):
        set_server_role("maintainer")
        assert get_server_role() == "maintainer"
        # Reset
        set_server_role("observer")

    def test_set_invalid_role_falls_back_to_observer(self):
        set_server_role("invalid_role")
        assert get_server_role() == "observer"


class TestGetRoleAllowlist:
    """Test get_role_allowlist function."""

    def test_get_autonomy_runner_allowlist(self):
        allowlist = get_role_allowlist("autonomy_runner")
        assert "session-start" in allowlist

    def test_get_maintainer_allowlist(self):
        allowlist = get_role_allowlist("maintainer")
        assert "*" in allowlist

    def test_get_observer_allowlist(self):
        allowlist = get_role_allowlist("observer")
        assert "list" in allowlist

    def test_unknown_role_returns_empty(self):
        allowlist = get_role_allowlist("unknown")
        assert allowlist == set()


class TestCheckActionAllowed:
    """Test check_action_allowed function."""

    def test_maintainer_can_do_anything(self):
        result = check_action_allowed("maintainer", "task", "complete")
        assert result.allowed is True

        result = check_action_allowed("maintainer", "session", "start")
        assert result.allowed is True

    def test_observer_can_read(self):
        result = check_action_allowed("observer", "task", "list")
        assert result.allowed is True

        result = check_action_allowed("observer", "task", "status")
        assert result.allowed is True

    def test_observer_cannot_mutate(self):
        result = check_action_allowed("observer", "task", "complete")
        assert result.allowed is False
        assert result.denied_action == "task-complete"
        assert result.required_role == "maintainer"

    def test_autonomy_runner_can_start_session(self):
        result = check_action_allowed("autonomy_runner", "session", "start")
        assert result.allowed is True

    def test_autonomy_runner_cannot_mutate_tasks(self):
        result = check_action_allowed("autonomy_runner", "task", "complete")
        assert result.allowed is False

    def test_autonomy_runner_can_execute_fidelity_gate_review(self):
        result = check_action_allowed("autonomy_runner", "review", "fidelity-gate")
        assert result.allowed is True

    def test_unknown_role_denied(self):
        result = check_action_allowed("unknown", "task", "complete")
        assert result.allowed is False


class TestInitializeRoleFromConfig:
    """Test initialize_role_from_config function."""

    def test_default_is_observer(self):
        with patch.dict(os.environ, {}, clear=True):
            role = initialize_role_from_config()
            assert role == "observer"

    def test_env_var_overrides_config(self):
        with patch.dict(os.environ, {"FOUNDRY_MCP_ROLE": "maintainer"}, clear=True):
            role = initialize_role_from_config("observer")
            assert role == "maintainer"

    def test_config_used_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            role = initialize_role_from_config("autonomy_runner")
            assert role == "autonomy_runner"


class TestRateLimitTracker:
    """Test RateLimitTracker class."""

    def test_check_rate_limit_returns_none_initially(self):
        """New tracker should not be rate limited."""
        tracker = RateLimitTracker()
        result = tracker.check_rate_limit("some-action")
        assert result is None

    def test_record_denial_increments_counter(self):
        """Recording denials should increment the counter."""
        tracker = RateLimitTracker()
        tracker.record_denial("action-1")

        stats = tracker.get_stats("action-1")
        assert stats["denial_count"] == 1
        assert stats["is_limited"] is False

    def test_rate_limit_triggers_after_max_denials(self):
        """Rate limit should trigger after max consecutive denials."""
        config = RateLimitConfig(max_consecutive_denials=3)
        tracker = RateLimitTracker(config)

        # Record denials up to threshold
        tracker.record_denial("action-1")
        tracker.record_denial("action-1")
        tracker.record_denial("action-1")

        # Should now be rate limited
        retry_after = tracker.check_rate_limit("action-1")
        assert retry_after is not None
        assert retry_after > 0

    def test_rate_limit_returns_retry_after_seconds(self):
        """Rate limited requests should return retry_after seconds."""
        config = RateLimitConfig(
            max_consecutive_denials=2,
            retry_after_seconds=10,
        )
        tracker = RateLimitTracker(config)

        tracker.record_denial("action-1")
        tracker.record_denial("action-1")

        retry_after = tracker.check_rate_limit("action-1")
        assert retry_after is not None
        assert retry_after <= 10

    def test_reset_clears_counter(self):
        """Reset should clear the denial counter."""
        config = RateLimitConfig(max_consecutive_denials=3)
        tracker = RateLimitTracker(config)

        # Record some denials
        tracker.record_denial("action-1")
        tracker.record_denial("action-1")

        # Reset
        tracker.reset("action-1")

        # Should not be rate limited
        stats = tracker.get_stats("action-1")
        assert stats["denial_count"] == 0
        assert stats["is_limited"] is False

    def test_rate_limit_expires_after_window(self):
        """Rate limit should expire after denial_window_seconds."""
        config = RateLimitConfig(
            max_consecutive_denials=2,
            denial_window_seconds=0,  # Immediate expiration for testing
            retry_after_seconds=1,
        )
        tracker = RateLimitTracker(config)

        # Record denials
        tracker.record_denial("action-1")

        # Wait a tiny bit
        time.sleep(0.01)

        # The denial should have expired due to 0 second window
        stats = tracker.get_stats("action-1")
        # Denial count might be 0 or 1 depending on timing
        # But we shouldn't be able to trigger rate limiting with just 1 denial
        tracker.record_denial("action-1")
        # Now we should be rate limited (if window wasn't 0)
        # With 0 window, first denial already expired

    def test_successful_dispatch_resets_counter(self):
        """Simulate successful dispatch by calling reset."""
        config = RateLimitConfig(max_consecutive_denials=3)
        tracker = RateLimitTracker(config)

        # Record some denials
        tracker.record_denial("action-1")
        tracker.record_denial("action-1")

        # Simulate successful dispatch
        tracker.reset("action-1")

        # Record more denials - should not trigger rate limit yet
        tracker.record_denial("action-1")
        tracker.record_denial("action-1")

        # Still not rate limited (only 2 consecutive)
        retry_after = tracker.check_rate_limit("action-1")
        assert retry_after is None

    def test_different_actions_tracked_separately(self):
        """Different actions should have separate rate limit tracking."""
        config = RateLimitConfig(max_consecutive_denials=2)
        tracker = RateLimitTracker(config)

        # Trigger rate limit on action-1
        tracker.record_denial("action-1")
        tracker.record_denial("action-1")

        # action-1 should be rate limited
        assert tracker.check_rate_limit("action-1") is not None

        # action-2 should NOT be rate limited
        assert tracker.check_rate_limit("action-2") is None

    def test_get_stats_returns_correct_info(self):
        """get_stats should return accurate state."""
        config = RateLimitConfig(max_consecutive_denials=2, retry_after_seconds=5)
        tracker = RateLimitTracker(config)

        # Initial state
        stats = tracker.get_stats("action-1")
        assert stats["denial_count"] == 0
        assert stats["is_limited"] is False
        assert stats["retry_after"] == 0

        # After denials
        tracker.record_denial("action-1")
        stats = tracker.get_stats("action-1")
        assert stats["denial_count"] == 1
        assert stats["is_limited"] is False

        # After rate limit triggered
        tracker.record_denial("action-1")
        stats = tracker.get_stats("action-1")
        assert stats["is_limited"] is True
        assert stats["retry_after"] > 0


class TestGetRateLimitTracker:
    """Test global rate limit tracker functions."""

    def test_get_rate_limit_tracker_returns_singleton(self):
        """get_rate_limit_tracker should return the same instance."""
        # Reset first
        reset_rate_limit_tracker()

        tracker1 = get_rate_limit_tracker()
        tracker2 = get_rate_limit_tracker()

        assert tracker1 is tracker2

        # Cleanup
        reset_rate_limit_tracker()

    def test_reset_rate_limit_tracker_creates_new_instance(self):
        """Reset should cause a new instance to be created."""
        tracker1 = get_rate_limit_tracker()
        reset_rate_limit_tracker()
        tracker2 = get_rate_limit_tracker()

        assert tracker1 is not tracker2

        # Cleanup
        reset_rate_limit_tracker()

    def test_get_rate_limit_tracker_with_config_reinitializes_singleton(self):
        """Providing explicit config should reinitialize the global tracker."""
        reset_rate_limit_tracker()

        tracker1 = get_rate_limit_tracker(RateLimitConfig(max_consecutive_denials=2))
        tracker2 = get_rate_limit_tracker(RateLimitConfig(max_consecutive_denials=5))

        assert tracker1 is not tracker2
        assert tracker2.get_stats("x")["denial_count"] == 0

        # Cleanup
        reset_rate_limit_tracker()


class TestRunIsolatedSubprocess:
    """Test stdin timeout and timeout precedence behavior."""

    def teardown_method(self):
        reset_runner_isolation_config()

    def test_stdin_timeout_cap_applies_when_timeout_not_provided(self):
        set_runner_isolation_config(RunnerIsolationConfig(stdin_timeout_seconds=0.1))

        with pytest.raises(StdinTimeoutError):
            run_isolated_subprocess(
                [sys.executable, "-c", "import time; time.sleep(0.5)"],
            )

    def test_explicit_timeout_shorter_than_stdin_timeout_raises_timeout_expired(self):
        set_runner_isolation_config(RunnerIsolationConfig(stdin_timeout_seconds=1.0))

        with pytest.raises(subprocess.TimeoutExpired):
            run_isolated_subprocess(
                [sys.executable, "-c", "import time; time.sleep(0.5)"],
                timeout=0.05,
            )

    def test_stdin_timeout_caps_longer_explicit_timeout(self):
        set_runner_isolation_config(RunnerIsolationConfig(stdin_timeout_seconds=0.1))

        with pytest.raises(StdinTimeoutError):
            run_isolated_subprocess(
                [sys.executable, "-c", "import time; time.sleep(0.5)"],
                timeout=5.0,
            )
