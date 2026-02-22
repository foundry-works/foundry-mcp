"""Telemetry invariant tests for unified tool routers.

Guards against regressions in metric names, request_id inclusion, and
details inclusion after the Phase 2 migration to shared helpers in common.py.

Each test is driven by a declarative baseline mapping so that adding a new
router requires only one table entry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Baseline: expected telemetry properties per router
# ---------------------------------------------------------------------------

# (router_module, tool_name, metric_prefix, has_local_request_id, include_details_in_router_error)
#
# - tool_name: the string passed to dispatch_with_standard_errors
# - metric_prefix: prefix used by the local _metric_name/_metric helper, or
#   None if the router has no local metric helper
# - has_local_request_id: True if the router defines a local _request_id()
# - include_details_in_router_error: True if the router passes this flag to
#   dispatch_with_standard_errors for ActionRouterError envelopes

ROUTER_BASELINES = [
    # module_name        tool_name       metric_prefix          has_rid  details
    ("authoring", "authoring", "authoring", True, False),
    ("environment", "environment", "environment", True, False),
    ("error", "error", None, False, False),
    ("health", "health", None, False, True),
    ("journal", "journal", None, False, False),
    ("lifecycle", "lifecycle", "lifecycle", True, False),
    ("plan", "plan", None, False, False),
    ("provider", "provider", "provider", True, False),
    ("research", "research", None, False, True),
    ("review", "review", None, False, False),
    ("server", "server", "unified_tools.server", True, False),
    ("spec", "spec", None, False, False),
    ("task", "task", "unified_tools.task", True, False),
    ("verification", "verification", "verification", True, False),
]

# Convenience sets derived from the baseline table
_ROUTERS_WITH_REQUEST_ID = {m for m, _, _, has_rid, _ in ROUTER_BASELINES if has_rid}
_ROUTERS_WITH_DETAILS = {m for m, _, _, _, details in ROUTER_BASELINES if details}
_ROUTERS_WITH_METRIC_PREFIX = {m: prefix for m, _, prefix, _, _ in ROUTER_BASELINES if prefix is not None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_dispatch(module_name: str):
    """Import the _dispatch_*_action function from a router module."""
    mod = __import__(
        f"foundry_mcp.tools.unified.{module_name}",
        fromlist=[f"_dispatch_{module_name}_action"],
    )
    return getattr(mod, f"_dispatch_{module_name}_action")


def _import_local_helper(module_name: str, helper_name: str):
    """Import a module-level helper from a router module, or None."""
    mod = __import__(
        f"foundry_mcp.tools.unified.{module_name}",
        fromlist=[helper_name],
    )
    return getattr(mod, helper_name, None)


# Dispatch function signatures vary: some take (action, payload), some take
# (*, action, payload, config), some take (action, **kwargs).  Build the
# minimal kwargs that satisfy each signature.
_DISPATCH_SIGNATURES: dict[str, dict] = {
    # (action=..., payload=..., config=...)
    "authoring": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "environment": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "error": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "journal": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "lifecycle": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "provider": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "review": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "server": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "spec": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "task": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    "verification": {"action": "nonexistent-action", "payload": {}, "config": MagicMock(specs_dir=None)},
    # Positional-style: (action, payload)
    "plan": {"action": "nonexistent-action", "payload": {}},
    # Unique signatures
    "health": {"action": "nonexistent-action"},
    "research": {"action": "nonexistent-action"},
}

# Dispatch functions that use **keyword-only** arguments
_KEYWORD_ONLY_DISPATCH = {
    "authoring",
    "environment",
    "error",
    "journal",
    "lifecycle",
    "provider",
    "review",
    "server",
    "spec",
    "task",
    "verification",
}

# Dispatch functions that use positional arguments


def _call_dispatch(module_name: str):
    """Call the dispatch function with the correct signature."""
    dispatch_fn = _import_dispatch(module_name)
    sig = _DISPATCH_SIGNATURES[module_name]
    if module_name in _KEYWORD_ONLY_DISPATCH:
        return dispatch_fn(**sig)
    elif module_name == "plan":
        return dispatch_fn(sig["action"], sig["payload"])
    elif module_name == "health":
        return dispatch_fn(sig["action"])
    elif module_name == "research":
        return dispatch_fn(sig["action"])
    else:
        return dispatch_fn(**sig)


# ---------------------------------------------------------------------------
# 1. Metric name invariants
# ---------------------------------------------------------------------------


class TestMetricNameInvariants:
    """Assert metric prefixes per router match the Phase 1 baseline."""

    @pytest.mark.parametrize(
        "module_name, expected_prefix",
        list(_ROUTERS_WITH_METRIC_PREFIX.items()),
        ids=list(_ROUTERS_WITH_METRIC_PREFIX.keys()),
    )
    def test_metric_prefix_matches_baseline(self, module_name, expected_prefix):
        """Each router's local metric helper produces the expected prefix."""
        # Some routers use _metric, others _metric_name
        fn = _import_local_helper(module_name, "_metric_name")
        helper_name = "_metric_name"
        if fn is None:
            fn = _import_local_helper(module_name, "_metric")
            helper_name = "_metric"
        assert fn is not None, f"{module_name} should define _metric_name() or _metric()"
        result = fn("test-action")
        assert result == f"{expected_prefix}.test_action", (
            f"{module_name}.{helper_name}('test-action') should produce '{expected_prefix}.test_action', got '{result}'"
        )

    @pytest.mark.parametrize(
        "module_name",
        [m for m, _, prefix, _, _ in ROUTER_BASELINES if prefix is None],
        ids=[m for m, _, prefix, _, _ in ROUTER_BASELINES if prefix is None],
    )
    def test_routers_without_local_metric_helper(self, module_name):
        """Routers without local metric helpers should not define _metric_name or _metric."""
        fn_metric_name = _import_local_helper(module_name, "_metric_name")
        fn_metric = _import_local_helper(module_name, "_metric")
        assert fn_metric_name is None and fn_metric is None, f"{module_name} should NOT define a local metric helper"


# ---------------------------------------------------------------------------
# 2. request_id invariants
# ---------------------------------------------------------------------------


class TestRequestIdInvariants:
    """Assert request_id inclusion per router matches the Phase 1 baseline."""

    @pytest.mark.parametrize(
        "module_name",
        sorted(_ROUTERS_WITH_REQUEST_ID),
        ids=sorted(_ROUTERS_WITH_REQUEST_ID),
    )
    def test_router_has_local_request_id(self, module_name):
        """Routers that had request_id in Phase 1 still define _request_id()."""
        fn = _import_local_helper(module_name, "_request_id")
        assert fn is not None, f"{module_name} should define _request_id()"

    @pytest.mark.parametrize(
        "module_name",
        sorted(_ROUTERS_WITH_REQUEST_ID),
        ids=sorted(_ROUTERS_WITH_REQUEST_ID),
    )
    def test_request_id_delegates_to_build_request_id(self, module_name):
        """Each _request_id() delegates to build_request_id with correct tool name."""
        # Find the expected tool_name from baseline
        tool_name = next(tn for m, tn, _, has_rid, _ in ROUTER_BASELINES if m == module_name and has_rid)
        with patch(
            f"foundry_mcp.tools.unified.{module_name}.build_request_id",
            return_value=f"{tool_name}_mock123",
        ) as mock_build:
            fn = _import_local_helper(module_name, "_request_id")
            result = fn()
            mock_build.assert_called_once_with(tool_name)
            assert result == f"{tool_name}_mock123"

    @pytest.mark.parametrize(
        "module_name",
        sorted({m for m, _, _, _, _ in ROUTER_BASELINES} - _ROUTERS_WITH_REQUEST_ID),
        ids=sorted({m for m, _, _, _, _ in ROUTER_BASELINES} - _ROUTERS_WITH_REQUEST_ID),
    )
    def test_router_does_not_have_local_request_id(self, module_name):
        """Routers without request_id in Phase 1 should not define _request_id()."""
        fn = _import_local_helper(module_name, "_request_id")
        assert fn is None, f"{module_name} should NOT define _request_id()"

    def test_request_id_router_count(self):
        """Exactly 7 routers define local _request_id helpers."""
        assert len(_ROUTERS_WITH_REQUEST_ID) == 7, (
            f"Expected 7 routers with _request_id, got {len(_ROUTERS_WITH_REQUEST_ID)}: "
            f"{sorted(_ROUTERS_WITH_REQUEST_ID)}"
        )


# ---------------------------------------------------------------------------
# 3. details inclusion invariants
# ---------------------------------------------------------------------------


class TestDetailsInclusionInvariants:
    """Assert include_details_in_router_error per router matches baseline."""

    @pytest.mark.parametrize(
        "module_name",
        sorted(_ROUTERS_WITH_DETAILS),
        ids=sorted(_ROUTERS_WITH_DETAILS),
    )
    def test_router_includes_details_in_router_error(self, module_name):
        """Routers flagged for details should include them in unsupported-action errors."""
        result = _call_dispatch(module_name)
        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        assert "details" in result["data"], f"{module_name} should include 'details' in ActionRouterError envelope"
        assert "allowed_actions" in result["data"]["details"], f"{module_name} details should contain 'allowed_actions'"

    @pytest.mark.parametrize(
        "module_name",
        sorted({m for m, _, _, _, _ in ROUTER_BASELINES} - _ROUTERS_WITH_DETAILS),
        ids=sorted({m for m, _, _, _, _ in ROUTER_BASELINES} - _ROUTERS_WITH_DETAILS),
    )
    def test_router_excludes_details_in_router_error(self, module_name):
        """Routers not flagged for details should not include them in unsupported-action errors."""
        result = _call_dispatch(module_name)
        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"
        # details should be None / absent for routers without the flag
        details = result["data"].get("details")
        assert details is None, (
            f"{module_name} should NOT include 'details' in ActionRouterError envelope, got: {details}"
        )

    def test_details_router_count(self):
        """Exactly 2 routers include details in ActionRouterError envelopes."""
        assert len(_ROUTERS_WITH_DETAILS) == 2, (
            f"Expected 2 routers with include_details_in_router_error, "
            f"got {len(_ROUTERS_WITH_DETAILS)}: {sorted(_ROUTERS_WITH_DETAILS)}"
        )

    def test_details_routers_are_health_and_research(self):
        """The details routers are exactly health and research."""
        assert _ROUTERS_WITH_DETAILS == {"health", "research"}


# ---------------------------------------------------------------------------
# 4. Dispatch tool_name invariants
# ---------------------------------------------------------------------------


class TestDispatchToolNameInvariants:
    """Assert each router passes the correct tool_name to dispatch_with_standard_errors."""

    @pytest.mark.parametrize(
        "module_name, expected_tool_name",
        [(m, tn) for m, tn, _, _, _ in ROUTER_BASELINES],
        ids=[m for m, _, _, _, _ in ROUTER_BASELINES],
    )
    def test_dispatch_tool_name_in_error_message(self, module_name, expected_tool_name):
        """Unsupported action error message should reference the correct tool_name."""
        result = _call_dispatch(module_name)
        assert result["success"] is False
        assert expected_tool_name in result["error"], (
            f"{module_name} dispatch error should mention tool_name '{expected_tool_name}', got: {result['error']}"
        )

    def test_all_14_routers_covered(self):
        """Baseline table covers all 14 routers."""
        assert len(ROUTER_BASELINES) == 14, f"Expected 14 router baselines, got {len(ROUTER_BASELINES)}"
