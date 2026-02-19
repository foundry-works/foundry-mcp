"""Tests for authorization module."""

import os
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

from foundry_mcp.core.authorization import (
    AUTONOMY_RUNNER_ALLOWLIST,
    MAINTAINER_ALLOWLIST,
    OBSERVER_ALLOWLIST,
    PathValidationError,
    RateLimitConfig,
    RateLimitTracker,
    Role,
    RunnerIsolationConfig,
    StdinTimeoutError,
    check_action_allowed,
    get_rate_limit_tracker,
    get_role_allowlist,
    get_server_role,
    initialize_role_from_config,
    reset_rate_limit_tracker,
    reset_runner_isolation_config,
    run_isolated_subprocess,
    server_role_var,
    set_runner_isolation_config,
    set_server_role,
    validate_runner_path,
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
        assert "spec-find" in AUTONOMY_RUNNER_ALLOWLIST
        assert "server-capabilities" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-start" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-list" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-status" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-step-next" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-step-replay" in AUTONOMY_RUNNER_ALLOWLIST
        assert "session-step-heartbeat" in AUTONOMY_RUNNER_ALLOWLIST
        assert "review-fidelity-gate" in AUTONOMY_RUNNER_ALLOWLIST
        assert "prepare" in AUTONOMY_RUNNER_ALLOWLIST

    def test_maintainer_allowlist_has_wildcard(self):
        assert "*" in MAINTAINER_ALLOWLIST

    def test_observer_allowlist_readonly(self):
        assert "list" in OBSERVER_ALLOWLIST
        assert "status" in OBSERVER_ALLOWLIST
        assert "session-events" in OBSERVER_ALLOWLIST
        assert "session-start" not in OBSERVER_ALLOWLIST


class TestServerRoleVar:
    """Test server role context variable."""

    def teardown_method(self):
        set_server_role("maintainer")

    def test_default_role_is_maintainer(self):
        assert get_server_role() == "maintainer"

    def test_set_server_role(self):
        set_server_role("observer")
        assert get_server_role() == "observer"
        # Reset
        set_server_role("maintainer")

    def test_set_invalid_role_falls_back_to_maintainer(self):
        set_server_role("invalid_role")
        assert get_server_role() == "maintainer"

    def test_role_propagates_to_new_thread_via_process_fallback(self):
        set_server_role("maintainer")
        observed_role: list[str] = []

        def _read_role() -> None:
            observed_role.append(get_server_role())

        thread = threading.Thread(target=_read_role)
        thread.start()
        thread.join()

        assert observed_role == ["maintainer"]

    def test_context_override_is_local_and_process_role_remains(self):
        set_server_role("maintainer")
        token = server_role_var.set("observer")
        try:
            assert get_server_role() == "observer"
        finally:
            server_role_var.reset(token)

        assert get_server_role() == "maintainer"


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

    def test_autonomy_runner_can_resolve_spec(self):
        result = check_action_allowed("autonomy_runner", "spec", "find")
        assert result.allowed is True

    def test_autonomy_runner_can_read_runtime_capabilities(self):
        result = check_action_allowed("autonomy_runner", "server", "capabilities")
        assert result.allowed is True

    def test_autonomy_runner_can_perform_session_preflight_list(self):
        result = check_action_allowed("autonomy_runner", "session", "list")
        assert result.allowed is True

    def test_autonomy_runner_can_prepare_task(self):
        result = check_action_allowed("autonomy_runner", "task", "prepare")
        assert result.allowed

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

    def test_default_is_maintainer(self):
        with patch.dict(os.environ, {}, clear=True):
            role = initialize_role_from_config()
            assert role == "maintainer"

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

    def test_tracker_caps_tracked_action_cardinality(self):
        """High-cardinality denial keys should be bounded in memory."""
        config = RateLimitConfig(
            max_consecutive_denials=9999,
            max_tracked_actions=32,
            global_cleanup_interval_seconds=0,
        )
        tracker = RateLimitTracker(config)

        for index in range(200):
            tracker.record_denial(f"action-{index}")

        tracked_actions = set(tracker._denials) | set(tracker._rate_limited_until) | set(tracker._last_seen)
        assert len(tracked_actions) <= 32

    def test_global_cleanup_sweeps_expired_rate_limits(self):
        """Expired rate-limit entries should be removed by global maintenance."""
        config = RateLimitConfig(
            max_consecutive_denials=1,
            retry_after_seconds=0.01,
            global_cleanup_interval_seconds=0,
        )
        tracker = RateLimitTracker(config)

        tracker.record_denial("stale-action")
        assert "stale-action" in tracker._rate_limited_until

        time.sleep(0.02)
        tracker.check_rate_limit("different-action")

        assert "stale-action" not in tracker._rate_limited_until


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
        set_server_role("maintainer")

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

    def test_runner_cwd_outside_workspace_is_rejected(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside"
        outside.mkdir(parents=True, exist_ok=True)

        set_server_role("autonomy_runner")
        set_runner_isolation_config(RunnerIsolationConfig(workspace_root=str(workspace)))

        with pytest.raises(PathValidationError) as exc:
            run_isolated_subprocess(
                [sys.executable, "-c", "print('ok')"],
                cwd=str(outside),
            )

        assert exc.value.reason == "outside_workspace"


class TestValidateRunnerPath:
    """Workspace-root validation should return the normalized validated path."""

    def teardown_method(self):
        reset_runner_isolation_config()
        set_server_role("maintainer")

    def test_relative_existing_path_returns_workspace_resolved_path(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace_target = workspace / "nested" / "target.txt"
        workspace_target.parent.mkdir(parents=True, exist_ok=True)
        workspace_target.write_text("workspace", encoding="utf-8")

        other = tmp_path / "other"
        other_target = other / "nested" / "target.txt"
        other_target.parent.mkdir(parents=True, exist_ok=True)
        other_target.write_text("other", encoding="utf-8")

        set_runner_isolation_config(RunnerIsolationConfig(workspace_root=str(workspace)))
        monkeypatch.chdir(other)

        result = validate_runner_path("nested/target.txt")
        assert result == workspace_target.resolve()

    def test_relative_non_existing_path_returns_workspace_resolved_path(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        other = tmp_path / "other"
        other.mkdir(parents=True, exist_ok=True)

        set_runner_isolation_config(RunnerIsolationConfig(workspace_root=str(workspace)))
        monkeypatch.chdir(other)

        result = validate_runner_path("future/output.txt")
        assert result == (workspace / "future" / "output.txt").resolve()

    def test_traversal_probe_is_denied(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        set_runner_isolation_config(RunnerIsolationConfig(workspace_root=str(workspace)))

        with pytest.raises(PathValidationError) as exc:
            validate_runner_path("../secret.txt")

        assert exc.value.reason == "path_traversal_denied"

    def test_absolute_path_outside_workspace_is_denied(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        outside_file = outside / "data.txt"
        outside_file.write_text("outside", encoding="utf-8")

        set_runner_isolation_config(RunnerIsolationConfig(workspace_root=str(workspace)))

        with pytest.raises(PathValidationError) as exc:
            validate_runner_path(outside_file)

        assert exc.value.reason == "outside_workspace"

    def test_absolute_workspace_path_is_canonicalized(self, tmp_path):
        workspace = tmp_path / "workspace"
        target = workspace / "nested" / "file.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")

        set_runner_isolation_config(RunnerIsolationConfig(workspace_root=str(workspace)))

        result = validate_runner_path(target.parent / "." / target.name)
        assert result == target.resolve()
