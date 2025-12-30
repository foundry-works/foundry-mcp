"""Research workflow implementations.

This package provides the workflow classes for multi-model orchestration:
- ChatWorkflow: Single-model conversation with thread persistence
- ConsensusWorkflow: Multi-model parallel consultation with synthesis
- ThinkDeepWorkflow: Hypothesis-driven systematic investigation
- IdeateWorkflow: Creative brainstorming with idea clustering
"""

from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase
from foundry_mcp.core.research.workflows.chat import ChatWorkflow
from foundry_mcp.core.research.workflows.consensus import ConsensusWorkflow
from foundry_mcp.core.research.workflows.thinkdeep import ThinkDeepWorkflow
from foundry_mcp.core.research.workflows.ideate import IdeateWorkflow

__all__ = [
    "ResearchWorkflowBase",
    "ChatWorkflow",
    "ConsensusWorkflow",
    "ThinkDeepWorkflow",
    "IdeateWorkflow",
]
