"""IDEATE workflow models (creative brainstorming)."""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from foundry_mcp.core.research.models.enums import IdeationPhase


class Idea(BaseModel):
    """A single idea generated in IDEATE workflow."""

    id: str = Field(default_factory=lambda: f"idea-{uuid4().hex[:8]}")
    content: str = Field(..., description="The idea content")
    perspective: Optional[str] = Field(default=None, description="Perspective that generated this idea")
    score: Optional[float] = Field(default=None, description="Score from 0-1 based on criteria")
    cluster_id: Optional[str] = Field(default=None, description="ID of cluster this idea belongs to")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(default=None)
    model_used: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdeaCluster(BaseModel):
    """A cluster of related ideas in IDEATE workflow."""

    id: str = Field(default_factory=lambda: f"cluster-{uuid4().hex[:8]}")
    name: str = Field(..., description="Cluster name/theme")
    description: Optional[str] = Field(default=None, description="Cluster description")
    idea_ids: list[str] = Field(default_factory=list, description="IDs of ideas in cluster")
    average_score: Optional[float] = Field(default=None)
    selected_for_elaboration: bool = Field(default=False)
    elaboration: Optional[str] = Field(default=None, description="Detailed elaboration if selected")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdeationState(BaseModel):
    """State for an IDEATE brainstorming session."""

    id: str = Field(default_factory=lambda: f"ideation-{uuid4().hex[:12]}")
    topic: str = Field(..., description="The topic being brainstormed")
    phase: IdeationPhase = Field(default=IdeationPhase.DIVERGENT)
    perspectives: list[str] = Field(default_factory=lambda: ["technical", "creative", "practical", "visionary"])
    ideas: list[Idea] = Field(default_factory=list)
    clusters: list[IdeaCluster] = Field(default_factory=list)
    scoring_criteria: list[str] = Field(default_factory=lambda: ["novelty", "feasibility", "impact"])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    system_prompt: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_idea(
        self,
        content: str,
        perspective: Optional[str] = None,
        **kwargs: Any,
    ) -> Idea:
        """Add a new idea to the session."""
        idea = Idea(content=content, perspective=perspective, **kwargs)
        self.ideas.append(idea)
        self.updated_at = datetime.utcnow()
        return idea

    def create_cluster(self, name: str, description: Optional[str] = None) -> IdeaCluster:
        """Create a new idea cluster."""
        cluster = IdeaCluster(name=name, description=description)
        self.clusters.append(cluster)
        self.updated_at = datetime.utcnow()
        return cluster

    def assign_idea_to_cluster(self, idea_id: str, cluster_id: str) -> bool:
        """Assign an idea to a cluster."""
        idea = next((i for i in self.ideas if i.id == idea_id), None)
        cluster = next((c for c in self.clusters if c.id == cluster_id), None)

        if idea and cluster:
            idea.cluster_id = cluster_id
            if idea_id not in cluster.idea_ids:
                cluster.idea_ids.append(idea_id)
            self.updated_at = datetime.utcnow()
            return True
        return False

    def advance_phase(self) -> IdeationPhase:
        """Advance to the next ideation phase."""
        phase_order = list(IdeationPhase)
        current_index = phase_order.index(self.phase)
        if current_index < len(phase_order) - 1:
            self.phase = phase_order[current_index + 1]
        self.updated_at = datetime.utcnow()
        return self.phase
