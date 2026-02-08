"""Deep Research workflow package.

Backward-compatible re-export shim. All public symbols remain importable
from ``foundry_mcp.core.research.workflows.deep_research`` exactly as before
the monolith was decomposed into a package.
"""

# Re-export everything from the monolith (transitional)
from foundry_mcp.core.research.workflows.deep_research._monolith import *  # noqa: F401,F403

# Explicit re-exports for symbols that tests/code import directly.
# Some symbols now live in extracted modules rather than _monolith.
from foundry_mcp.core.research.workflows.deep_research._monolith import (  # noqa: F401
    # Core workflow
    DeepResearchWorkflow,
    # Classes patched at this module path by tests (test_deep_research_digest.py)
    ContentSummarizer,
    DocumentDigestor,
    PDFExtractor,
    ContextBudgetManager,
)
from foundry_mcp.core.research.workflows.deep_research.orchestration import (  # noqa: F401
    AgentRole,
    AgentDecision,
    SupervisorHooks,
    SupervisorOrchestrator,
)
from foundry_mcp.core.research.workflows.deep_research.infrastructure import (  # noqa: F401
    _active_research_sessions,
    _active_sessions_lock,
    _crash_handler,
    _cleanup_on_exit,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import (  # noqa: F401
    get_domain_quality,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (  # noqa: F401
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
)
