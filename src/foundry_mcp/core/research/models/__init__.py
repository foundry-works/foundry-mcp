"""Research workflow models package.

Re-exports all public symbols for backward compatibility.
Callers can continue using:
    from foundry_mcp.core.research.models import X, Y, Z
"""

# --- Wave 1: Extracted sub-modules ---
# --- Wave 2: Extracted sub-modules ---
from foundry_mcp.core.research.models.consensus import (
    ConsensusConfig,
    ConsensusState,
    ModelResponse,
)
from foundry_mcp.core.research.models.conversations import (
    ConversationMessage,
    ConversationThread,
)

# --- Wave 3: Extracted sub-modules ---
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchConfig,
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.digest import (
    DigestPayload,
    EvidenceSnippet,
    get_base_id,
    is_fragment_id,
    make_fragment_id,
    parse_fragment_id,
)
from foundry_mcp.core.research.models.enums import (
    ConfidenceLevel,
    ConsensusStrategy,
    IdeationPhase,
    ThreadStatus,
    WorkflowType,
)
from foundry_mcp.core.research.models.fidelity import (
    ContentFidelityRecord,
    FidelityLevel,
    PhaseContentFidelityRecord,
    PhaseMetrics,
)
from foundry_mcp.core.research.models.ideation import (
    Idea,
    IdeaCluster,
    IdeationState,
)
from foundry_mcp.core.research.models.sources import (
    DOMAIN_TIERS,
    ResearchFinding,
    ResearchGap,
    ResearchMode,
    ResearchSource,
    SourceQuality,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.models.thinkdeep import (
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)

__all__ = [
    # Fragment utilities
    "make_fragment_id",
    "parse_fragment_id",
    "is_fragment_id",
    "get_base_id",
    # Digest models
    "EvidenceSnippet",
    "DigestPayload",
    # Shared enums
    "WorkflowType",
    "ConfidenceLevel",
    "ConsensusStrategy",
    "ThreadStatus",
    "IdeationPhase",
    # Conversation models
    "ConversationMessage",
    "ConversationThread",
    # THINKDEEP models
    "Hypothesis",
    "InvestigationStep",
    "ThinkDeepState",
    # IDEATE models
    "Idea",
    "IdeaCluster",
    "IdeationState",
    # CONSENSUS models
    "ModelResponse",
    "ConsensusConfig",
    "ConsensusState",
    # Deep research config
    "DeepResearchConfig",
    "DeepResearchPhase",
    # Fidelity models
    "FidelityLevel",
    "PhaseContentFidelityRecord",
    "ContentFidelityRecord",
    "PhaseMetrics",
    # Source models
    "SourceType",
    "SourceQuality",
    "ResearchMode",
    "DOMAIN_TIERS",
    "SubQuery",
    "ResearchSource",
    "ResearchFinding",
    "ResearchGap",
    # Deep research state
    "DeepResearchState",
]
