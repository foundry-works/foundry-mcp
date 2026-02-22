"""Shared parametrized dispatch contract tests for all unified tool routers.

Consolidates the per-router dispatch error envelope tests previously spread
across individual test classes into a single data-driven suite.  Each router
is described by a baseline entry; the tests are generated via parametrize.

Also includes full-envelope snapshot tests for representative routers to
catch message/detail regressions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Router dispatch baseline
# ---------------------------------------------------------------------------
#
# Each entry: (
#   module_name,        -- e.g. "authoring"
#   dispatch_fn_name,   -- e.g. "_dispatch_authoring_action"
#   router_const_name,  -- e.g. "_AUTHORING_ROUTER"
#   tool_name,          -- string passed to dispatch_with_standard_errors
#   call_style,         -- "kw" (keyword-only) | "pos" (positional) | "health" | "research"
#   valid_action,       -- a real action name for the internal-error test
# )

DISPATCH_BASELINES = [
    ("authoring", "_dispatch_authoring_action", "_AUTHORING_ROUTER", "authoring", "kw", "create"),
    ("environment", "_dispatch_environment_action", "_ENVIRONMENT_ROUTER", "environment", "kw", "info"),
    ("error", "_dispatch_error_action", "_ERROR_ROUTER", "error", "kw", "list"),
    ("health", "_dispatch_health_action", "_HEALTH_ROUTER", "health", "health", "check"),
    ("journal", "_dispatch_journal_action", "_JOURNAL_ROUTER", "journal", "kw", "add"),
    ("lifecycle", "_dispatch_lifecycle_action", "_LIFECYCLE_ROUTER", "lifecycle", "kw", "move"),
    ("plan", "_dispatch_plan_action", "_PLAN_ROUTER", "plan", "pos", "create"),
    ("provider", "_dispatch_provider_action", "_PROVIDER_ROUTER", "provider", "kw", "list"),
    ("research_handlers", "_dispatch_research_action", "_RESEARCH_ROUTER", "research", "research", "chat"),
    ("review", "_dispatch_review_action", "_REVIEW_ROUTER", "review", "kw", "spec"),
    ("server", "_dispatch_server_action", "_SERVER_ROUTER", "server", "kw", "tools"),
    ("spec", "_dispatch_spec_action", "_SPEC_ROUTER", "spec", "kw", "list"),
    ("task", "_dispatch_task_action", "_TASK_ROUTER", "task", "kw", "list"),
    ("verification", "_dispatch_verification_action", "_VERIFICATION_ROUTER", "verification", "kw", "add"),
]

# Derive IDs for parametrize
_BASELINE_IDS = [entry[0] for entry in DISPATCH_BASELINES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config(mock_config):
    """Extend base mock_config with specs_dir=None for dispatch tests."""
    mock_config.specs_dir = None
    return mock_config


def _import(module_name: str, attr: str):
    """Import *attr* from ``foundry_mcp.tools.unified.<module_name>``."""
    mod = __import__(
        f"foundry_mcp.tools.unified.{module_name}",
        fromlist=[attr],
    )
    return getattr(mod, attr)


def _call_dispatch(module_name, dispatch_fn_name, call_style, action, mock_config):
    """Call a router dispatch function with the correct signature."""
    fn = _import(module_name, dispatch_fn_name)
    if call_style == "kw":
        return fn(action=action, payload={}, config=mock_config)
    elif call_style == "pos":
        return fn(action, {}, config=mock_config)
    elif call_style == "health":
        return fn(action=action, config=mock_config)
    elif call_style == "research":
        return fn(action=action)
    else:
        raise ValueError(f"Unknown call_style: {call_style}")


# ---------------------------------------------------------------------------
# Envelope assertion helpers (shared with snapshot tests)
# ---------------------------------------------------------------------------


def assert_error_envelope(response: dict) -> None:
    """Assert response-v2 error envelope invariants."""
    assert isinstance(response, dict), "Response must be a dict"
    assert response["success"] is False
    assert isinstance(response["error"], str) and response["error"]
    assert isinstance(response["data"], dict)
    assert isinstance(response["meta"], dict)
    assert response["meta"]["version"] == "response-v2"
    assert "error_code" in response["data"]
    assert "error_type" in response["data"]


def assert_unsupported_action_envelope(response: dict) -> None:
    """Assert envelope for unsupported action errors."""
    assert_error_envelope(response)
    assert "unsupported" in response["error"].lower()
    assert response["data"]["error_code"] == "VALIDATION_ERROR"
    assert response["data"]["error_type"] == "validation"


def assert_internal_error_envelope(response: dict) -> None:
    """Assert envelope for internal/unexpected errors."""
    assert_error_envelope(response)
    assert response["data"]["error_code"] == "INTERNAL_ERROR"
    assert response["data"]["error_type"] == "internal"
    assert "details" in response["data"]
    assert "action" in response["data"]["details"]
    assert "error_type" in response["data"]["details"]


# ---------------------------------------------------------------------------
# 1. Parametrized unsupported-action tests (all 14 routers)
# ---------------------------------------------------------------------------


class TestUnsupportedActionEnvelope:
    """Every router produces a valid VALIDATION_ERROR envelope for unknown actions."""

    @pytest.mark.parametrize(
        "module_name, dispatch_fn_name, router_const, tool_name, call_style, valid_action",
        DISPATCH_BASELINES,
        ids=_BASELINE_IDS,
    )
    def test_unsupported_action(
        self,
        mock_config,
        module_name,
        dispatch_fn_name,
        router_const,
        tool_name,
        call_style,
        valid_action,
    ):
        result = _call_dispatch(
            module_name,
            dispatch_fn_name,
            call_style,
            "nonexistent-action",
            mock_config,
        )
        assert_unsupported_action_envelope(result)
        # Error message references the tool name
        assert tool_name in result["error"]

    def test_all_14_routers_covered(self):
        assert len(DISPATCH_BASELINES) == 14


# ---------------------------------------------------------------------------
# 2. Parametrized internal-error tests (all 14 routers)
# ---------------------------------------------------------------------------


class TestInternalErrorEnvelope:
    """Every router produces a valid INTERNAL_ERROR envelope for unexpected exceptions."""

    @pytest.mark.parametrize(
        "module_name, dispatch_fn_name, router_const, tool_name, call_style, valid_action",
        DISPATCH_BASELINES,
        ids=_BASELINE_IDS,
    )
    def test_internal_error(
        self,
        mock_config,
        module_name,
        dispatch_fn_name,
        router_const,
        tool_name,
        call_style,
        valid_action,
    ):
        patch_target = f"foundry_mcp.tools.unified.{module_name}.{router_const}"
        with (
            patch(patch_target) as mock_router,
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="maintainer",
            ),
        ):
            mock_router.allowed_actions.return_value = [valid_action]
            mock_router.dispatch.side_effect = RuntimeError("boom")
            result = _call_dispatch(
                module_name,
                dispatch_fn_name,
                call_style,
                valid_action,
                mock_config,
            )
        assert_internal_error_envelope(result)
        # Details capture the action and exception type
        assert result["data"]["details"]["action"] == valid_action
        assert result["data"]["details"]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# 3. Authorization parity tests (all 14 routers)
# ---------------------------------------------------------------------------


class TestAuthorizationParity:
    """Every router should enforce authorization consistently."""

    @pytest.mark.parametrize(
        "module_name, dispatch_fn_name, router_const, tool_name, call_style, valid_action",
        DISPATCH_BASELINES,
        ids=_BASELINE_IDS,
    )
    def test_router_enforces_authorization(
        self,
        mock_config,
        module_name,
        dispatch_fn_name,
        router_const,
        tool_name,
        call_style,
        valid_action,
    ):
        patch_target = f"foundry_mcp.tools.unified.{module_name}.{router_const}"
        with (
            patch(patch_target) as mock_router,
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="observer",
            ),
            patch(
                "foundry_mcp.tools.unified.common.get_rate_limit_tracker",
            ) as mock_tracker_factory,
            patch(
                "foundry_mcp.tools.unified.common.check_action_allowed",
            ) as mock_check_action_allowed,
        ):
            mock_router.allowed_actions.return_value = [valid_action]
            mock_router.dispatch.return_value = {"success": True, "data": {}}

            mock_tracker = MagicMock()
            mock_tracker.check_rate_limit.return_value = None
            mock_tracker_factory.return_value = mock_tracker
            mock_check_action_allowed.return_value = MagicMock(
                allowed=False,
                required_role="maintainer",
            )

            result = _call_dispatch(
                module_name,
                dispatch_fn_name,
                call_style,
                valid_action,
                mock_config,
            )

        assert_error_envelope(result)
        assert result["data"]["error_code"] == "AUTHORIZATION"
        mock_router.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Full-envelope snapshot tests for representative routers
# ---------------------------------------------------------------------------


class TestEnvelopeSnapshots:
    """Full-envelope structure checks for representative routers.

    These catch regressions in message wording, details shape, and metadata
    fields that parametrized structural tests might miss.
    """

    def test_environment_unsupported_action_snapshot(self, mock_config):
        """Environment: full envelope for unsupported action."""
        result = _call_dispatch(
            "environment",
            "_dispatch_environment_action",
            "kw",
            "nonexistent-action",
            mock_config,
        )
        # Structure
        assert result["success"] is False
        assert result["meta"]["version"] == "response-v2"
        assert isinstance(result["meta"]["request_id"], str)
        assert len(result["meta"]["request_id"]) > 0
        # Error details
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        assert result["data"]["error_type"] == "validation"
        assert "environment" in result["error"]
        assert "nonexistent-action" in result["error"]
        assert "Allowed actions" in result["error"] or "allowed" in result["error"].lower()
        # Remediation present
        assert isinstance(result["data"].get("remediation"), str)

    def test_health_unsupported_action_snapshot_with_details(self):
        """Health: full envelope includes details for unsupported action."""
        result = _call_dispatch(
            "health",
            "_dispatch_health_action",
            "health",
            "nonexistent-action",
            None,
        )
        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        # Health uses include_details_in_router_error=True
        details = result["data"]["details"]
        assert "action" in details
        assert details["action"] == "nonexistent-action"
        assert "allowed_actions" in details
        assert isinstance(details["allowed_actions"], list)
        assert len(details["allowed_actions"]) > 0

    def test_server_internal_error_snapshot(self, mock_config):
        """Server: full envelope for internal error."""
        with (
            patch("foundry_mcp.tools.unified.server._SERVER_ROUTER") as mock_router,
            patch(
                "foundry_mcp.tools.unified.common.get_server_role",
                return_value="maintainer",
            ),
        ):
            mock_router.allowed_actions.return_value = ["tools"]
            mock_router.dispatch.side_effect = ValueError("db connection lost")
            result = _call_dispatch(
                "server",
                "_dispatch_server_action",
                "kw",
                "tools",
                mock_config,
            )
        assert result["success"] is False
        assert result["meta"]["version"] == "response-v2"
        assert result["data"]["error_code"] == "INTERNAL_ERROR"
        assert result["data"]["error_type"] == "internal"
        # Message includes the error text
        assert "db connection lost" in result["error"]
        # Details capture action and exception type
        assert result["data"]["details"]["action"] == "tools"
        assert result["data"]["details"]["error_type"] == "ValueError"
        # Remediation present
        assert isinstance(result["data"].get("remediation"), str)

    def test_research_unsupported_action_snapshot_with_details(self):
        """Research: full envelope includes details for unsupported action."""
        result = _call_dispatch(
            "research",
            "_dispatch_research_action",
            "research",
            "nonexistent-action",
            None,
        )
        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        # Research uses include_details_in_router_error=True
        details = result["data"]["details"]
        assert details["action"] == "nonexistent-action"
        assert isinstance(details["allowed_actions"], list)

    def test_task_internal_error_snapshot(self, mock_config):
        """Task: full envelope for internal error with empty exception message."""
        with patch("foundry_mcp.tools.unified.task._TASK_ROUTER") as mock_router:
            mock_router.allowed_actions.return_value = ["list"]
            mock_router.dispatch.side_effect = RuntimeError()
            result = _call_dispatch(
                "task",
                "_dispatch_task_action",
                "kw",
                "list",
                mock_config,
            )
        assert result["success"] is False
        assert result["data"]["error_code"] == "INTERNAL_ERROR"
        # Empty message falls back to class name
        assert "RuntimeError" in result["error"]
        assert result["data"]["details"]["error_type"] == "RuntimeError"
