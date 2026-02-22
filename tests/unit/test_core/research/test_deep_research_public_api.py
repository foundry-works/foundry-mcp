"""Contract tests for the deep_research package public API.

Verifies backward-compatible re-exports, no circular imports, correct MRO,
and that every submodule is importable in isolation.
"""

import importlib
import sys

import pytest


def test_all_public_symbols_importable_from_original_path():
    """Verify backward-compat re-exports from the original import path."""
    from foundry_mcp.core.research.workflows.deep_research import (  # noqa: F401
        ANALYSIS_OUTPUT_RESERVED,
        ANALYSIS_PHASE_BUDGET_FRACTION,
        REFINEMENT_OUTPUT_RESERVED,
        REFINEMENT_PHASE_BUDGET_FRACTION,
        SYNTHESIS_OUTPUT_RESERVED,
        SYNTHESIS_PHASE_BUDGET_FRACTION,
        AgentDecision,
        AgentRole,
        DeepResearchWorkflow,
        SupervisorHooks,
        SupervisorOrchestrator,
        # These are underscore-prefixed but explicitly re-exported in __init__.py
        # for internal use by infrastructure and tests. We verify re-export
        # availability here, not endorsement for external consumption.
        _active_research_sessions,
        _active_sessions_lock,
        get_domain_quality,
    )


def test_patched_classes_importable_from_package():
    """Verify classes patched by tests are re-exported at package level."""
    from foundry_mcp.core.research.workflows.deep_research import (  # noqa: F401
        ContentSummarizer,
        ContextBudgetManager,
        DocumentDigestor,
        PDFExtractor,
    )


def test_old_monolith_path_not_importable():
    """Verify the retired _monolith module cannot be imported.

    After the Stage 6 rename (_monolith.py -> core.py), the old path must
    not resolve. This prevents accidental resurrection of the shim.
    """
    # Ensure it's not cached from a prior test run
    mod_path = "foundry_mcp.core.research.workflows.deep_research._monolith"
    sys.modules.pop(mod_path, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod_path)


def test_no_circular_imports():
    """Verify no circular import errors in the deep_research package.

    Imports every submodule in dependency order. If the import graph has a
    cycle, one of these will raise ImportError.
    """
    _PKG = "foundry_mcp.core.research.workflows.deep_research"
    # Leaf modules first, then composite modules that depend on them
    for suffix in (
        "._constants",
        "._helpers",
        "._budgeting",
        ".infrastructure",
        ".source_quality",
        ".orchestration",
        ".phases.planning",
        ".phases.gathering",
        ".phases.analysis",
        ".phases.synthesis",
        ".phases.refinement",
        ".background_tasks",
        ".session_management",
        ".core",
        "",  # __init__.py (re-exports everything)
    ):
        importlib.import_module(f"{_PKG}{suffix}")


def test_workflow_inherits_all_phase_methods():
    """Verify the mixin MRO provides all expected phase methods."""
    from foundry_mcp.core.research.workflows.deep_research import (
        DeepResearchWorkflow,
    )

    expected_methods = [
        "_execute_planning_async",
        "_execute_gathering_async",
        "_execute_analysis_async",
        "_execute_synthesis_async",
        "_execute_refinement_async",
        "list_sessions",
        "delete_session",
        "resume_research",
        "_start_background_task",
        "get_background_task",
        "cleanup_stale_tasks",
    ]
    for method in expected_methods:
        assert hasattr(DeepResearchWorkflow, method), f"Missing method: {method}"


@pytest.mark.parametrize(
    "module",
    [
        "foundry_mcp.core.research.workflows.deep_research.core",
        "foundry_mcp.core.research.workflows.deep_research.orchestration",
        "foundry_mcp.core.research.workflows.deep_research.infrastructure",
        "foundry_mcp.core.research.workflows.deep_research.source_quality",
        "foundry_mcp.core.research.workflows.deep_research._constants",
        "foundry_mcp.core.research.workflows.deep_research._helpers",
        "foundry_mcp.core.research.workflows.deep_research._budgeting",
        "foundry_mcp.core.research.workflows.deep_research.background_tasks",
        "foundry_mcp.core.research.workflows.deep_research.session_management",
        "foundry_mcp.core.research.workflows.deep_research.phases.planning",
        "foundry_mcp.core.research.workflows.deep_research.phases.gathering",
        "foundry_mcp.core.research.workflows.deep_research.phases.analysis",
        "foundry_mcp.core.research.workflows.deep_research.phases.synthesis",
        "foundry_mcp.core.research.workflows.deep_research.phases.refinement",
    ],
)
def test_submodule_importable(module):
    """Verify every submodule imports without errors."""
    importlib.import_module(module)
