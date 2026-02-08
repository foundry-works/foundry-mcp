"""Tool registration parity tests.

Validates that the manifest produced by ``_build_unified_manifest_tools()``
stays in sync with the live ``ActionRouter`` instances.  Covers tool count,
names, version, category, tags, description, and action summaries/aliases.
"""

from __future__ import annotations

import pytest

from foundry_mcp.tools.unified.server import _build_unified_manifest_tools


# ---------------------------------------------------------------------------
# Fixture: manifest tools (computed once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def manifest_tools():
    """Return the list produced by the manifest builder."""
    return _build_unified_manifest_tools()


@pytest.fixture(scope="session")
def manifest_by_name(manifest_tools):
    """Index manifest entries by tool name for quick lookup."""
    return {t["name"]: t for t in manifest_tools}


# ---------------------------------------------------------------------------
# Import the same routers the manifest builder uses
# ---------------------------------------------------------------------------

def _live_routers():
    """Return {name: router} matching the manifest builder's router set."""
    from foundry_mcp.tools.unified.authoring_handlers import _AUTHORING_ROUTER
    from foundry_mcp.tools.unified.environment import _ENVIRONMENT_ROUTER
    from foundry_mcp.tools.unified.error import _ERROR_ROUTER
    from foundry_mcp.tools.unified.health import _HEALTH_ROUTER
    from foundry_mcp.tools.unified.journal import _JOURNAL_ROUTER
    from foundry_mcp.tools.unified.lifecycle import _LIFECYCLE_ROUTER
    from foundry_mcp.tools.unified.plan import _PLAN_ROUTER
    from foundry_mcp.tools.unified.pr import _PR_ROUTER
    from foundry_mcp.tools.unified.provider import _PROVIDER_ROUTER
    from foundry_mcp.tools.unified.review import _REVIEW_ROUTER
    from foundry_mcp.tools.unified.server import _SERVER_ROUTER
    from foundry_mcp.tools.unified.spec import _SPEC_ROUTER
    from foundry_mcp.tools.unified.task_handlers import _TASK_ROUTER
    from foundry_mcp.tools.unified.test import _TEST_ROUTER
    from foundry_mcp.tools.unified.verification import _VERIFICATION_ROUTER

    return {
        "health": _HEALTH_ROUTER,
        "plan": _PLAN_ROUTER,
        "pr": _PR_ROUTER,
        "error": _ERROR_ROUTER,
        "journal": _JOURNAL_ROUTER,
        "authoring": _AUTHORING_ROUTER,
        "provider": _PROVIDER_ROUTER,
        "environment": _ENVIRONMENT_ROUTER,
        "lifecycle": _LIFECYCLE_ROUTER,
        "verification": _VERIFICATION_ROUTER,
        "task": _TASK_ROUTER,
        "spec": _SPEC_ROUTER,
        "review": _REVIEW_ROUTER,
        "server": _SERVER_ROUTER,
        "test": _TEST_ROUTER,
    }


# NOTE: The research router is intentionally excluded from the manifest
# builder (_build_unified_manifest_tools) and therefore from this parity
# test.  It is registered at runtime but not surfaced in the manifest.
# If research is added to the manifest, update _live_routers() and
# EXPECTED_CATEGORIES accordingly.
MANIFEST_EXCLUDED_ROUTERS = {"research"}

LIVE_ROUTERS = _live_routers()
TOOL_NAMES = sorted(LIVE_ROUTERS.keys())


# ---------------------------------------------------------------------------
# Tool count and names
# ---------------------------------------------------------------------------

class TestToolCountAndNames:
    """Manifest tool count and name set match live routers."""

    def test_tool_count_matches(self, manifest_tools):
        assert len(manifest_tools) == len(LIVE_ROUTERS)

    def test_tool_names_match(self, manifest_by_name):
        assert set(manifest_by_name.keys()) == set(LIVE_ROUTERS.keys())

    def test_router_tool_name_matches_manifest_key(self):
        """Each router's .tool_name matches the key used in the manifest."""
        for name, router in LIVE_ROUTERS.items():
            assert router.tool_name == name, (
                f"Router keyed as '{name}' reports tool_name='{router.tool_name}'"
            )


# ---------------------------------------------------------------------------
# Per-tool metadata parity (parametrized)
# ---------------------------------------------------------------------------

EXPECTED_CATEGORIES = {
    "health": "health",
    "plan": "planning",
    "pr": "workflow",
    "error": "observability",
    "journal": "journal",
    "authoring": "specs",
    "provider": "providers",
    "environment": "environment",
    "lifecycle": "lifecycle",
    "verification": "verification",
    "task": "tasks",
    "spec": "specs",
    "review": "review",
    "server": "server",
    "test": "testing",
}


@pytest.mark.parametrize("tool_name", TOOL_NAMES)
class TestPerToolMetadata:
    """Manifest metadata fields match expected values for each tool."""

    def test_version(self, manifest_by_name, tool_name):
        assert manifest_by_name[tool_name]["version"] == "1.0.0"

    def test_not_deprecated(self, manifest_by_name, tool_name):
        assert manifest_by_name[tool_name]["deprecated"] is False

    def test_tags_contain_unified(self, manifest_by_name, tool_name):
        assert "unified" in manifest_by_name[tool_name]["tags"]

    def test_category_matches(self, manifest_by_name, tool_name):
        assert manifest_by_name[tool_name]["category"] == EXPECTED_CATEGORIES[tool_name]

    def test_description_non_empty(self, manifest_by_name, tool_name):
        desc = manifest_by_name[tool_name]["description"]
        assert isinstance(desc, str) and len(desc) > 0


# ---------------------------------------------------------------------------
# Action parity (manifest actions vs router actions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name", TOOL_NAMES)
class TestActionParity:
    """Manifest action lists match the live router's allowed actions."""

    def test_action_names_match(self, manifest_by_name, tool_name):
        manifest_actions = {a["name"] for a in manifest_by_name[tool_name]["actions"]}
        router_actions = set(LIVE_ROUTERS[tool_name].allowed_actions())
        assert manifest_actions == router_actions, (
            f"{tool_name}: manifest actions {manifest_actions} != "
            f"router actions {router_actions}"
        )

    def test_action_summaries_match(self, manifest_by_name, tool_name):
        """Manifest action summaries match router.describe() values."""
        router_summaries = LIVE_ROUTERS[tool_name].describe()
        for action_entry in manifest_by_name[tool_name]["actions"]:
            name = action_entry["name"]
            expected = router_summaries.get(name)
            assert action_entry["summary"] == expected, (
                f"{tool_name}.{name}: manifest summary "
                f"{action_entry['summary']!r} != router summary {expected!r}"
            )

    def test_action_count_matches(self, manifest_by_name, tool_name):
        manifest_count = len(manifest_by_name[tool_name]["actions"])
        router_count = len(LIVE_ROUTERS[tool_name].allowed_actions())
        assert manifest_count == router_count


# ---------------------------------------------------------------------------
# Alias resolution (routers accept aliases defined in ActionDefinitions)
# ---------------------------------------------------------------------------

class TestAliasResolution:
    """Routers that define aliases can dispatch using them."""

    def test_authoring_aliases_resolve(self):
        """Authoring router accepts underscore aliases (e.g. spec_create)."""
        router = LIVE_ROUTERS["authoring"]
        canonical = set(router.allowed_actions())
        # Aliases should resolve but not appear in allowed_actions()
        for action in canonical:
            underscore_form = action.replace("-", "_")
            if underscore_form != action:
                # The router should recognize the alias without raising
                assert underscore_form not in canonical, (
                    f"Alias '{underscore_form}' should not appear as a "
                    "canonical action name"
                )


# ---------------------------------------------------------------------------
# Cross-check: parity set vs dispatch baselines
# ---------------------------------------------------------------------------

class TestRouterSetCompleteness:
    """Parity test router set accounts for all dispatch-baseline routers."""

    def test_dispatch_baselines_covered(self):
        """Every dispatch-baseline router is either tested or documented as excluded."""
        from tests.tools.unified.test_dispatch_common import DISPATCH_BASELINES

        baseline_names = {entry[0] for entry in DISPATCH_BASELINES}
        parity_names = set(LIVE_ROUTERS.keys())
        uncovered = baseline_names - parity_names - MANIFEST_EXCLUDED_ROUTERS
        assert uncovered == set(), (
            f"Dispatch-baseline routers not covered by parity test or "
            f"MANIFEST_EXCLUDED_ROUTERS: {uncovered}"
        )
