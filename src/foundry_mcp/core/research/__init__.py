"""Research workflows for multi-model orchestration.

This package provides conversation threading, multi-model consensus,
hypothesis-driven investigation, and creative brainstorming workflows.
"""

from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    ConsensusConfig,
    ConsensusState,
    ConsensusStrategy,
    ConversationMessage,
    ConversationThread,
    Hypothesis,
    Idea,
    IdeaCluster,
    IdeationPhase,
    IdeationState,
    InvestigationStep,
    ModelResponse,
    ThreadStatus,
    ThinkDeepState,
    WorkflowType,
)
from foundry_mcp.core.research.memory import (
    FileStorageBackend,
    ResearchMemory,
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
