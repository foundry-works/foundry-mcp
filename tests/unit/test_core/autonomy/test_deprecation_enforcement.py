"""Tests for deprecation removal target enforcement (M2).

Covers:
- Unexpired action passes through
- Expired action returns hard error
- Env var escape hatch bypasses enforcement
"""

import os
from unittest.mock import patch

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _check_deprecation_expired,
)


class TestCheckDeprecationExpired:
    """_check_deprecation_expired enforcement logic."""

    def test_unexpired_returns_none(self):
        deprecation = {
            "action": "session-start",
            "replacement": 'task(action="session", command="start")',
            "removal_target": "2099-12-31_or_2_minor_releases",
        }
        result = _check_deprecation_expired(deprecation, request_id="req-1")
        assert result is None

    def test_expired_returns_hard_error(self):
        deprecation = {
            "action": "session-start",
            "replacement": 'task(action="session", command="start")',
            "removal_target": "2020-01-01_or_2_minor_releases",
        }
        result = _check_deprecation_expired(deprecation, request_id="req-1")
        assert result is not None
        assert result["success"] is False
        assert "removed on 2020-01-01" in result["error"]
        assert 'task(action="session", command="start")' in result["error"]

    def test_env_var_escape_hatch_bypasses(self):
        deprecation = {
            "action": "session-start",
            "replacement": 'task(action="session", command="start")',
            "removal_target": "2020-01-01_or_2_minor_releases",
        }
        with patch.dict(os.environ, {"FOUNDRY_MCP_ALLOW_DEPRECATED_ACTIONS": "true"}):
            result = _check_deprecation_expired(deprecation, request_id="req-1")
        assert result is None

    def test_unparseable_target_fails_open(self):
        deprecation = {
            "action": "session-start",
            "replacement": 'task(action="session", command="start")',
            "removal_target": "not-a-date",
        }
        result = _check_deprecation_expired(deprecation, request_id="req-1")
        assert result is None

    def test_empty_removal_target_returns_none(self):
        deprecation = {
            "action": "session-start",
            "replacement": 'task(action="session", command="start")',
            "removal_target": "",
        }
        result = _check_deprecation_expired(deprecation, request_id="req-1")
        assert result is None
