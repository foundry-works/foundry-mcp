"""Tests for write-lock enforcement helpers.

Covers:
- Protected vs read-only action classification
- Lock enforcement for non-terminal sessions
- Bypass with reason logging
- Terminal session status derivation from canonical enum set
"""

from unittest.mock import patch

import pytest

from foundry_mcp.core.autonomy.models.enums import SessionStatus
from foundry_mcp.core.autonomy.write_lock import (
    NON_TERMINAL_SESSION_STATUSES,
    PROTECTED_LIFECYCLE_ACTIONS,
    PROTECTED_TASK_ACTIONS,
    READ_ONLY_TASK_ACTIONS,
    TERMINAL_SESSION_STATUSES,
    WriteLockResult,
    WriteLockStatus,
    check_and_enforce_write_lock,
    check_autonomy_write_lock,
    is_protected_action,
    make_write_lock_error_response,
)

# =============================================================================
# Action Classification
# =============================================================================


class TestIsProtectedAction:
    """Test is_protected_action correctly classifies actions."""

    @pytest.mark.parametrize("action", sorted(PROTECTED_TASK_ACTIONS))
    def test_protected_task_actions(self, action):
        assert is_protected_action(action, "task") is True

    @pytest.mark.parametrize("action", sorted(READ_ONLY_TASK_ACTIONS))
    def test_read_only_task_actions(self, action):
        assert is_protected_action(action, "task") is False

    @pytest.mark.parametrize("action", sorted(PROTECTED_LIFECYCLE_ACTIONS))
    def test_protected_lifecycle_actions(self, action):
        assert is_protected_action(action, "lifecycle") is True

    def test_unknown_lifecycle_action_is_not_protected(self):
        """Unknown lifecycle actions are NOT protected (not in the set)."""
        assert is_protected_action("info", "lifecycle") is False

    def test_unknown_task_action_defaults_to_protected(self):
        """Unknown task actions default to protected (conservative)."""
        assert is_protected_action("some-new-mutation", "task") is True

    def test_unknown_category_defaults_to_protected(self):
        assert is_protected_action("start", "unknown_category") is True

    def test_case_insensitive(self):
        assert is_protected_action("START", "task") is True
        assert is_protected_action("Info", "task") is False

    def test_empty_action_is_protected(self):
        assert is_protected_action("", "task") is True

    def test_none_action_name_is_protected(self):
        """None converted to empty string is protected."""
        # The function lowercases the string, None would error,
        # but let's check the empty string path
        assert is_protected_action("", "task") is True


# =============================================================================
# Terminal / Non-Terminal Status Sets
# =============================================================================


class TestStatusSets:
    """Test that terminal/non-terminal sets are derived from canonical enum."""

    def test_terminal_statuses_are_strings(self):
        for status in TERMINAL_SESSION_STATUSES:
            assert isinstance(status, str)

    def test_terminal_includes_completed_and_ended(self):
        assert "completed" in TERMINAL_SESSION_STATUSES
        assert "ended" in TERMINAL_SESSION_STATUSES

    def test_non_terminal_includes_running_and_paused(self):
        assert "running" in NON_TERMINAL_SESSION_STATUSES
        assert "paused" in NON_TERMINAL_SESSION_STATUSES

    def test_terminal_and_non_terminal_are_disjoint(self):
        assert TERMINAL_SESSION_STATUSES & NON_TERMINAL_SESSION_STATUSES == frozenset()

    def test_all_statuses_covered(self):
        all_values = {s.value for s in SessionStatus}
        assert TERMINAL_SESSION_STATUSES | NON_TERMINAL_SESSION_STATUSES == all_values


# =============================================================================
# Write Lock Check
# =============================================================================


class TestCheckAutonomyWriteLock:
    """Test check_autonomy_write_lock with mocked session discovery."""

    def test_no_active_session_returns_allowed(self):
        with patch(
            "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
            return_value=None,
        ):
            result = check_autonomy_write_lock("spec-1", "/workspace")
        assert result.status == WriteLockStatus.ALLOWED
        assert result.lock_active is False

    def test_active_session_returns_locked(self):
        session_data = {"id": "session-1", "status": "running"}
        with patch(
            "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
            return_value=session_data,
        ):
            result = check_autonomy_write_lock("spec-1", "/workspace")
        assert result.status == WriteLockStatus.LOCKED
        assert result.lock_active is True
        assert result.session_id == "session-1"

    def test_bypass_with_reason_returns_bypassed(self):
        session_data = {"id": "session-1", "status": "running"}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="maintainer",
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock._write_bypass_journal_entry",
                return_value=True,
            ),
        ):
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason="Emergency fix needed",
                allow_lock_bypass=True,  # Must be enabled for bypass to work
            )
        assert result.status == WriteLockStatus.BYPASSED
        assert result.lock_active is True
        assert result.bypass_logged is True

    def test_bypass_without_reason_stays_locked(self):
        session_data = {"id": "session-1", "status": "running"}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="maintainer",
            ),
        ):
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason=None,
                allow_lock_bypass=True,
            )
        assert result.status == WriteLockStatus.LOCKED
        assert result.metadata.get("error") == "bypass_reason_required"

    def test_bypass_with_empty_reason_stays_locked(self):
        session_data = {"id": "session-1", "status": "running"}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="maintainer",
            ),
        ):
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason="   ",
                allow_lock_bypass=True,
            )
        assert result.status == WriteLockStatus.LOCKED
        assert result.metadata.get("error") == "bypass_reason_required"

    def test_bypass_denied_for_non_maintainer_role(self):
        """Bypass is denied by role even when config allows bypass."""
        session_data = {"id": "session-1", "status": "running", "write_lock_enforced": True}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="observer",
            ),
        ):
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason="Emergency fix needed",
                allow_lock_bypass=True,
            )
        assert result.status == WriteLockStatus.LOCKED
        assert result.lock_active is True
        assert result.metadata.get("error") == "bypass_denied_role"

    def test_role_check_precedes_config_check(self):
        """Role-denied error takes precedence over config-denied error."""
        session_data = {"id": "session-1", "status": "running", "write_lock_enforced": True}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="observer",
            ),
        ):
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason="Emergency fix needed",
                allow_lock_bypass=False,
            )
        assert result.status == WriteLockStatus.LOCKED
        assert result.metadata.get("error") == "bypass_denied_role"

    def test_bypass_denied_by_config(self):
        """Bypass is denied when allow_lock_bypass=False (default)."""
        session_data = {"id": "session-1", "status": "running", "write_lock_enforced": True}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="maintainer",
            ),
        ):
            # Default allow_lock_bypass=False should deny bypass
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason="Emergency fix needed",
                allow_lock_bypass=False,  # Explicit default
            )
        assert result.status == WriteLockStatus.LOCKED
        assert result.lock_active is True
        assert result.metadata.get("error") == "bypass_denied_by_config"

    def test_bypass_allowed_when_config_permits(self):
        """Bypass works when allow_lock_bypass=True."""
        session_data = {"id": "session-1", "status": "running", "write_lock_enforced": True}
        with (
            patch(
                "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
                return_value=session_data,
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock.get_server_role",
                return_value="maintainer",
            ),
            patch(
                "foundry_mcp.core.autonomy.write_lock._write_bypass_journal_entry",
                return_value=True,
            ),
        ):
            result = check_autonomy_write_lock(
                "spec-1",
                "/workspace",
                bypass_flag=True,
                bypass_reason="Emergency fix needed",
                allow_lock_bypass=True,  # Explicitly enabled
            )
        assert result.status == WriteLockStatus.BYPASSED
        assert result.bypass_logged is True

    def test_write_lock_not_enforced_returns_allowed(self):
        session_data = {
            "id": "session-1",
            "status": "running",
            "write_lock_enforced": False,
        }
        with patch(
            "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
            return_value=session_data,
        ):
            result = check_autonomy_write_lock("spec-1", "/workspace")
        assert result.status == WriteLockStatus.ALLOWED
        assert result.lock_active is True  # Lock exists but not enforced


# =============================================================================
# check_and_enforce_write_lock (Convenience)
# =============================================================================


class TestCheckAndEnforceWriteLock:
    """Test the convenience enforcement function."""

    def test_read_only_action_always_allowed(self):
        """Read-only actions bypass lock check entirely."""
        result = check_and_enforce_write_lock(
            spec_id="spec-1",
            workspace="/workspace",
            action_name="info",
            action_category="task",
        )
        assert result is None  # No error = allowed

    def test_protected_action_blocked_when_locked(self):
        session_data = {"id": "session-1", "status": "running"}
        with patch(
            "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
            return_value=session_data,
        ):
            result = check_and_enforce_write_lock(
                spec_id="spec-1",
                workspace="/workspace",
                action_name="complete",
                action_category="task",
            )
        assert result is not None  # Error response returned
        assert result.success is False
        assert result.error is not None

    def test_protected_action_allowed_when_no_session(self):
        with patch(
            "foundry_mcp.core.autonomy.write_lock._find_active_session_for_spec",
            return_value=None,
        ):
            result = check_and_enforce_write_lock(
                spec_id="spec-1",
                workspace="/workspace",
                action_name="complete",
                action_category="task",
            )
        assert result is None


# =============================================================================
# Error Response
# =============================================================================


class TestMakeWriteLockErrorResponse:
    """Test error response construction for write lock violations."""

    def test_includes_session_id(self):
        lock_result = WriteLockResult(
            status=WriteLockStatus.LOCKED,
            lock_active=True,
            session_id="session-1",
            session_status="running",
        )
        response = make_write_lock_error_response(lock_result, action_name="complete")
        assert response.success is False
        assert "session-1" in response.error

    def test_includes_action_name_in_message(self):
        lock_result = WriteLockResult(
            status=WriteLockStatus.LOCKED,
            lock_active=True,
            session_id="session-1",
            session_status="running",
        )
        response = make_write_lock_error_response(lock_result, action_name="delete")
        assert response.success is False
        assert "delete" in response.error
