"""Deep Research workflow package.

Backward-compatible re-export shim. All public symbols remain importable
from ``foundry_mcp.core.research.workflows.deep_research`` exactly as before
the monolith was decomposed into a package.
"""

# Re-export everything from the monolith (transitional)
from foundry_mcp.core.research.workflows.deep_research._monolith import *  # noqa: F401,F403

# Explicit re-exports for symbols that tests/code import directly:
from foundry_mcp.core.research.workflows.deep_research._monolith import (  # noqa: F401
    # Core workflow
    DeepResearchWorkflow,
    # Orchestration
    AgentRole,
    AgentDecision,
    SupervisorHooks,
    SupervisorOrchestrator,
    # Infrastructure / threading state
    _active_research_sessions,
    _active_sessions_lock,
    _crash_handler,
    _cleanup_on_exit,
    # Source quality
    get_domain_quality,
    # Constants
    ANALYSIS_PHASE_BUDGET_FRACTION,
    ANALYSIS_OUTPUT_RESERVED,
    SYNTHESIS_PHASE_BUDGET_FRACTION,
    SYNTHESIS_OUTPUT_RESERVED,
    REFINEMENT_PHASE_BUDGET_FRACTION,
    REFINEMENT_OUTPUT_RESERVED,
    REFINEMENT_REPORT_BUDGET_FRACTION,
    FINAL_FIT_MAX_ITERATIONS,
    FINAL_FIT_COMPRESSION_FACTOR,
    FINAL_FIT_SAFETY_MARGIN,
    # Classes patched at this module path by tests (test_deep_research_digest.py)
    ContentSummarizer,
    DocumentDigestor,
    PDFExtractor,
    ContextBudgetManager,
)
