"""Tests for M4: audit logging for authorization denials."""

import logging

from foundry_mcp.core.authorization import check_action_allowed


class TestAuthorizationDenialAuditLogging:
    """Verify authorization denials emit structured warning logs."""

    def test_denial_logs_warning(self, caplog):
        """Denied action logs AUTHORIZATION_DENIED at WARNING level."""
        with caplog.at_level(logging.WARNING, logger="foundry_mcp.core.authorization"):
            result = check_action_allowed("observer", "task", "complete")
        assert not result.allowed
        assert any("AUTHORIZATION_DENIED" in msg for msg in caplog.messages)

    def test_allowed_does_not_log(self, caplog):
        """Allowed action does not emit AUTHORIZATION_DENIED."""
        with caplog.at_level(logging.WARNING, logger="foundry_mcp.core.authorization"):
            result = check_action_allowed("maintainer", "task", "complete")
        assert result.allowed
        assert not any("AUTHORIZATION_DENIED" in msg for msg in caplog.messages)

    def test_unknown_role_logs_warning(self, caplog):
        """Unknown role denial logs with reason=unknown_role."""
        with caplog.at_level(logging.WARNING, logger="foundry_mcp.core.authorization"):
            result = check_action_allowed("nonexistent_role", "task", "complete")
        assert not result.allowed
        assert any("unknown_role" in msg for msg in caplog.messages)
