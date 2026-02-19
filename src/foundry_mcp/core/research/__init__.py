"""Research workflows for multi-model orchestration.

This package provides conversation threading, multi-model consensus,
hypothesis-driven investigation, and creative brainstorming workflows.
"""

from foundry_mcp.core.research.memory import (
    FileStorageBackend,
    ResearchMemory,
)
from foundry_mcp.core.research.models.consensus import (
    ConsensusConfig,
    ConsensusState,
    ModelResponse,
)
from foundry_mcp.core.research.models.conversations import (
    ConversationMessage,
    ConversationThread,
)
from foundry_mcp.core.research.models.enums import (
    ConfidenceLevel,
    ConsensusStrategy,
    IdeationPhase,
    ThreadStatus,
    WorkflowType,
)
from foundry_mcp.core.research.models.ideation import (
    Idea,
    IdeaCluster,
    IdeationState,
)
from foundry_mcp.core.research.models.thinkdeep import (
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)
from foundry_mcp.core.research.workflows import (
    ChatWorkflow,
    ConsensusWorkflow,
    IdeateWorkflow,
    ResearchWorkflowBase,
    ThinkDeepWorkflow,
)

__all__ = [
    # Enums
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
    # Storage
    "FileStorageBackend",
    "ResearchMemory",
    # Workflows
    "ResearchWorkflowBase",
    "ChatWorkflow",
    "ConsensusWorkflow",
    "ThinkDeepWorkflow",
    "IdeateWorkflow",
]
