"""Deep Research workflow package.

All public symbols are re-exported here so that imports from
``foundry_mcp.core.research.workflows.deep_research`` continue to work.
"""

# Re-export everything from the core module (includes DeepResearchWorkflow)
from foundry_mcp.core.research.models.deep_research import DeepResearchState  # noqa: F401
from foundry_mcp.core.research.workflows.deep_research._budgeting import (  # noqa: F401
    ContextBudgetManager,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (  # noqa: F401
    ANALYSIS_OUTPUT_RESERVED,
    ANALYSIS_PHASE_BUDGET_FRACTION,
    FINAL_FIT_COMPRESSION_FACTOR,
    FINAL_FIT_MAX_ITERATIONS,
    FINAL_FIT_SAFETY_MARGIN,
    REFINEMENT_OUTPUT_RESERVED,
    REFINEMENT_PHASE_BUDGET_FRACTION,
    REFINEMENT_REPORT_BUDGET_FRACTION,
    SYNTHESIS_OUTPUT_RESERVED,
    SYNTHESIS_PHASE_BUDGET_FRACTION,
)
from foundry_mcp.core.research.workflows.deep_research.core import *  # noqa: F401,F403
from foundry_mcp.core.research.workflows.deep_research.infrastructure import (  # noqa: F401
    _active_research_sessions,
    _active_sessions_lock,
    _cleanup_on_exit,
    _crash_handler,
)
from foundry_mcp.core.research.workflows.deep_research.orchestration import (  # noqa: F401
    AgentDecision,
    AgentRole,
    SupervisorHooks,
    SupervisorOrchestrator,
)

# Explicit re-exports for symbols from extracted modules.
# These classes are patched at module paths by tests (test_deep_research_digest.py)
# and are re-exported from phases.analysis for backward compatibility.
from foundry_mcp.core.research.workflows.deep_research.phases.analysis import (  # noqa: F401
    ContentSummarizer,
    DocumentDigestor,
    PDFExtractor,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import (  # noqa: F401
    get_domain_quality,
)
