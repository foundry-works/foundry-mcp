"""Shared enums for research workflow models."""

from enum import Enum


class WorkflowType(str, Enum):
    """Types of research workflows available."""

    CHAT = "chat"
    CONSENSUS = "consensus"
    THINKDEEP = "thinkdeep"
    IDEATE = "ideate"
    DEEP_RESEARCH = "deep_research"


class ConfidenceLevel(str, Enum):
    """Confidence levels for hypotheses in THINKDEEP workflow."""

    SPECULATION = "speculation"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CONFIRMED = "confirmed"


class ConsensusStrategy(str, Enum):
    """Strategies for synthesizing multi-model responses in CONSENSUS workflow."""

    ALL_RESPONSES = "all_responses"  # Return all responses without synthesis
    SYNTHESIZE = "synthesize"  # Use a model to synthesize responses
    MAJORITY = "majority"  # Use majority vote for factual questions
    FIRST_VALID = "first_valid"  # Return first successful response


class ThreadStatus(str, Enum):
    """Status of a conversation thread."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class IdeationPhase(str, Enum):
    """Phases of the IDEATE workflow."""

    DIVERGENT = "divergent"  # Generate diverse ideas
    CONVERGENT = "convergent"  # Cluster and score ideas
    SELECTION = "selection"  # Select clusters for elaboration
    ELABORATION = "elaboration"  # Develop selected ideas
