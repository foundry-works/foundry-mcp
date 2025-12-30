"""IDEATE workflow for creative brainstorming with phased execution.

Provides creative ideation capabilities with multi-perspective generation,
idea clustering, scoring, and elaboration phases.
"""

import logging
from typing import Any, Optional

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    Idea,
    IdeaCluster,
    IdeationPhase,
    IdeationState,
)
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult

logger = logging.getLogger(__name__)


class IdeateWorkflow(ResearchWorkflowBase):
    """Creative brainstorming workflow with phased execution.

    Features:
    - Divergent phase: Multi-perspective idea generation
    - Convergent phase: Idea clustering and scoring
    - Selection phase: Mark clusters for elaboration
    - Elaboration phase: Develop selected clusters
    - Persistent state across sessions
    """

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
    ) -> None:
        """Initialize ideate workflow.

        Args:
            config: Research configuration
            memory: Optional memory instance
        """
        super().__init__(config, memory)

    def execute(
        self,
        topic: Optional[str] = None,
        ideation_id: Optional[str] = None,
        action: str = "generate",
        perspective: Optional[str] = None,
        cluster_ids: Optional[list[str]] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        perspectives: Optional[list[str]] = None,
        scoring_criteria: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute an ideation action.

        Args:
            topic: Topic for new ideation session
            ideation_id: Existing session to continue
            action: Action to perform (generate, cluster, score, select, elaborate)
            perspective: Specific perspective for idea generation
            cluster_ids: Cluster IDs for selection/elaboration
            system_prompt: System prompt for new sessions
            provider_id: Provider to use
            perspectives: Custom perspectives (uses config default if None)
            scoring_criteria: Custom scoring criteria

        Returns:
            WorkflowResult with ideation results
        """
        # Get or create state
        if ideation_id:
            state = self.memory.load_ideation(ideation_id)
            if not state:
                return WorkflowResult(
                    success=False,
                    content="",
                    error=f"Ideation session {ideation_id} not found",
                )
        elif topic:
            state = IdeationState(
                topic=topic,
                perspectives=perspectives or self.config.ideate_perspectives,
                scoring_criteria=scoring_criteria or ["novelty", "feasibility", "impact"],
                system_prompt=system_prompt,
            )
        else:
            return WorkflowResult(
                success=False,
                content="",
                error="Either 'topic' (for new session) or 'ideation_id' (to continue) is required",
            )

        # Dispatch to action handler
        if action == "generate":
            result = self._generate_ideas(state, perspective, provider_id)
        elif action == "cluster":
            result = self._cluster_ideas(state, provider_id)
        elif action == "score":
            result = self._score_ideas(state, provider_id)
        elif action == "select":
            result = self._select_clusters(state, cluster_ids)
        elif action == "elaborate":
            result = self._elaborate_clusters(state, provider_id)
        elif action == "status":
            result = self._get_status(state)
        else:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Unknown action '{action}'. Valid: generate, cluster, score, select, elaborate, status",
            )

        if result.success:
            # Persist state
            self.memory.save_ideation(state)

            # Add common metadata
            result.metadata["ideation_id"] = state.id
            result.metadata["phase"] = state.phase.value
            result.metadata["idea_count"] = len(state.ideas)
            result.metadata["cluster_count"] = len(state.clusters)

        return result

    def _generate_ideas(
        self,
        state: IdeationState,
        perspective: Optional[str],
        provider_id: Optional[str],
    ) -> WorkflowResult:
        """Generate ideas from a perspective.

        Args:
            state: Ideation state
            perspective: Perspective to generate from (or all if None)
            provider_id: Provider to use

        Returns:
            WorkflowResult with generated ideas
        """
        perspectives_to_use = [perspective] if perspective else state.perspectives

        all_ideas = []
        for persp in perspectives_to_use:
            prompt = self._build_generation_prompt(state.topic, persp)
            result = self._execute_provider(
                prompt=prompt,
                provider_id=provider_id,
                system_prompt=self._build_ideation_system_prompt(),
            )

            if result.success:
                # Parse ideas from response
                ideas = self._parse_ideas(result.content, persp, result.provider_id, result.model_used)
                for idea in ideas:
                    state.ideas.append(idea)
                    all_ideas.append(idea)

        if not all_ideas:
            return WorkflowResult(
                success=False,
                content="",
                error="No ideas generated",
            )

        # Format output
        content = f"Generated {len(all_ideas)} ideas:\n\n"
        for i, idea in enumerate(all_ideas, 1):
            content += f"{i}. [{idea.perspective}] {idea.content}\n"

        return WorkflowResult(
            success=True,
            content=content,
            metadata={
                "ideas_generated": len(all_ideas),
                "perspectives_used": perspectives_to_use,
            },
        )

    def _cluster_ideas(
        self,
        state: IdeationState,
        provider_id: Optional[str],
    ) -> WorkflowResult:
        """Cluster related ideas.

        Args:
            state: Ideation state
            provider_id: Provider to use

        Returns:
            WorkflowResult with clustering results
        """
        if not state.ideas:
            return WorkflowResult(
                success=False,
                content="",
                error="No ideas to cluster. Generate ideas first.",
            )

        # Build clustering prompt
        ideas_text = "\n".join(f"{i+1}. {idea.content}" for i, idea in enumerate(state.ideas))
        prompt = f"""Analyze these ideas and group them into 3-5 thematic clusters:

{ideas_text}

For each cluster, provide:
1. A short name (2-4 words)
2. A brief description
3. The idea numbers that belong to it

Format as:
CLUSTER: [name]
DESCRIPTION: [description]
IDEAS: [comma-separated numbers]"""

        result = self._execute_provider(
            prompt=prompt,
            provider_id=provider_id,
            system_prompt="You are organizing ideas into thematic clusters. Be systematic and comprehensive.",
        )

        if not result.success:
            return result

        # Parse clusters from response
        clusters = self._parse_clusters(result.content, state)

        # Update state
        state.clusters = clusters
        state.phase = IdeationPhase.CONVERGENT

        # Format output
        content = f"Created {len(clusters)} clusters:\n\n"
        for cluster in clusters:
            idea_count = len(cluster.idea_ids)
            content += f"**{cluster.name}** ({idea_count} ideas)\n{cluster.description}\n\n"

        return WorkflowResult(
            success=True,
            content=content,
            metadata={"clusters_created": len(clusters)},
        )

    def _score_ideas(
        self,
        state: IdeationState,
        provider_id: Optional[str],
    ) -> WorkflowResult:
        """Score ideas based on criteria.

        Args:
            state: Ideation state
            provider_id: Provider to use

        Returns:
            WorkflowResult with scoring results
        """
        if not state.ideas:
            return WorkflowResult(
                success=False,
                content="",
                error="No ideas to score.",
            )

        criteria_text = ", ".join(state.scoring_criteria)
        ideas_text = "\n".join(f"{i+1}. {idea.content}" for i, idea in enumerate(state.ideas))

        prompt = f"""Score each idea on a scale of 0.0 to 1.0 based on these criteria: {criteria_text}

Ideas:
{ideas_text}

Provide an overall score (average of criteria) for each idea.
Format: [idea number]: [score] - [brief justification]"""

        result = self._execute_provider(
            prompt=prompt,
            provider_id=provider_id,
            system_prompt="You are evaluating ideas systematically. Be fair and objective.",
        )

        if not result.success:
            return result

        # Parse scores from response
        self._parse_scores(result.content, state)

        # Update cluster scores
        for cluster in state.clusters:
            cluster_ideas = [i for i in state.ideas if i.id in cluster.idea_ids]
            if cluster_ideas:
                scores = [i.score for i in cluster_ideas if i.score is not None]
                if scores:
                    cluster.average_score = sum(scores) / len(scores)

        # Format output
        scored_ideas = [(i, i.score) for i in state.ideas if i.score is not None]
        scored_ideas.sort(key=lambda x: x[1] or 0, reverse=True)

        content = "Scored ideas (top to bottom):\n\n"
        for idea, score in scored_ideas[:10]:
            content += f"- {idea.content[:50]}... (score: {score:.2f})\n"

        return WorkflowResult(
            success=True,
            content=content,
            metadata={"ideas_scored": len(scored_ideas)},
        )

    def _select_clusters(
        self,
        state: IdeationState,
        cluster_ids: Optional[list[str]],
    ) -> WorkflowResult:
        """Select clusters for elaboration.

        Args:
            state: Ideation state
            cluster_ids: Cluster IDs to select

        Returns:
            WorkflowResult with selection confirmation
        """
        if not state.clusters:
            return WorkflowResult(
                success=False,
                content="",
                error="No clusters to select. Run clustering first.",
            )

        if not cluster_ids:
            # Auto-select top clusters by score
            sorted_clusters = sorted(
                state.clusters,
                key=lambda c: c.average_score or 0,
                reverse=True,
            )
            cluster_ids = [c.id for c in sorted_clusters[:2]]

        selected = []
        for cluster in state.clusters:
            if cluster.id in cluster_ids:
                cluster.selected_for_elaboration = True
                selected.append(cluster)

        if not selected:
            return WorkflowResult(
                success=False,
                content="",
                error=f"No matching clusters found for IDs: {cluster_ids}",
            )

        state.phase = IdeationPhase.SELECTION

        content = f"Selected {len(selected)} clusters for elaboration:\n\n"
        for cluster in selected:
            content += f"- **{cluster.name}**: {cluster.description}\n"

        return WorkflowResult(
            success=True,
            content=content,
            metadata={"selected_clusters": [c.id for c in selected]},
        )

    def _elaborate_clusters(
        self,
        state: IdeationState,
        provider_id: Optional[str],
    ) -> WorkflowResult:
        """Elaborate selected clusters into detailed plans.

        Args:
            state: Ideation state
            provider_id: Provider to use

        Returns:
            WorkflowResult with elaborations
        """
        selected = [c for c in state.clusters if c.selected_for_elaboration]

        if not selected:
            return WorkflowResult(
                success=False,
                content="",
                error="No clusters selected for elaboration.",
            )

        elaborations = []
        for cluster in selected:
            # Get ideas in cluster
            cluster_ideas = [i for i in state.ideas if i.id in cluster.idea_ids]
            ideas_text = "\n".join(f"- {i.content}" for i in cluster_ideas)

            prompt = f"""Elaborate on this cluster of ideas into a detailed plan:

Cluster: {cluster.name}
Description: {cluster.description}

Ideas in this cluster:
{ideas_text}

Provide:
1. A comprehensive synthesis of the ideas
2. Key implementation steps
3. Potential challenges and mitigations
4. Expected outcomes"""

            result = self._execute_provider(
                prompt=prompt,
                provider_id=provider_id,
                system_prompt="You are developing ideas into actionable plans. Be thorough and practical.",
            )

            if result.success:
                cluster.elaboration = result.content
                elaborations.append((cluster, result.content))

        state.phase = IdeationPhase.ELABORATION

        content = f"Elaborated {len(elaborations)} clusters:\n\n"
        for cluster, elab in elaborations:
            content += f"## {cluster.name}\n\n{elab}\n\n---\n\n"

        return WorkflowResult(
            success=True,
            content=content,
            metadata={"clusters_elaborated": len(elaborations)},
        )

    def _get_status(self, state: IdeationState) -> WorkflowResult:
        """Get current ideation status.

        Args:
            state: Ideation state

        Returns:
            WorkflowResult with status summary
        """
        content = f"""# Ideation Status: {state.topic}

**Phase**: {state.phase.value}
**Ideas**: {len(state.ideas)}
**Clusters**: {len(state.clusters)}
**Created**: {state.created_at.isoformat()}
**Updated**: {state.updated_at.isoformat()}

## Perspectives
{', '.join(state.perspectives)}

## Scoring Criteria
{', '.join(state.scoring_criteria)}
"""

        if state.clusters:
            content += "\n## Clusters\n"
            for cluster in state.clusters:
                selected = " [SELECTED]" if cluster.selected_for_elaboration else ""
                score = f" (score: {cluster.average_score:.2f})" if cluster.average_score else ""
                content += f"- {cluster.name}{score}{selected}\n"

        return WorkflowResult(
            success=True,
            content=content,
        )

    def _build_generation_prompt(self, topic: str, perspective: str) -> str:
        """Build idea generation prompt.

        Args:
            topic: Ideation topic
            perspective: Perspective to generate from

        Returns:
            Generation prompt
        """
        return f"""Generate 5-7 creative ideas for: {topic}

Approach this from a {perspective} perspective. Think freely and don't self-censor.

For each idea, provide a single sentence description.
Format: One idea per line, starting with a dash (-)"""

    def _build_ideation_system_prompt(self) -> str:
        """Build system prompt for ideation.

        Returns:
            System prompt
        """
        return """You are a creative brainstorming assistant. Generate diverse, innovative ideas without judgment.
Focus on quantity and variety - the evaluation comes later. Be bold and think outside the box."""

    def _parse_ideas(
        self,
        response: str,
        perspective: str,
        provider_id: Optional[str],
        model_used: Optional[str],
    ) -> list[Idea]:
        """Parse ideas from response.

        Args:
            response: Provider response
            perspective: Perspective used
            provider_id: Provider ID
            model_used: Model used

        Returns:
            List of parsed ideas
        """
        ideas = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                content = line[1:].strip()
                if content:
                    ideas.append(
                        Idea(
                            content=content,
                            perspective=perspective,
                            provider_id=provider_id,
                            model_used=model_used,
                        )
                    )
        return ideas

    def _parse_clusters(self, response: str, state: IdeationState) -> list[IdeaCluster]:
        """Parse clusters from response.

        Args:
            response: Provider response
            state: Ideation state

        Returns:
            List of parsed clusters
        """
        clusters = []
        current_name = None
        current_desc = None
        current_ideas = []

        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("CLUSTER:"):
                # Save previous cluster if exists
                if current_name:
                    cluster = IdeaCluster(name=current_name, description=current_desc)
                    cluster.idea_ids = current_ideas
                    clusters.append(cluster)
                current_name = line.split(":", 1)[1].strip()
                current_desc = None
                current_ideas = []
            elif line.upper().startswith("DESCRIPTION:"):
                current_desc = line.split(":", 1)[1].strip()
            elif line.upper().startswith("IDEAS:"):
                # Parse idea numbers
                nums_str = line.split(":", 1)[1].strip()
                for num in nums_str.replace(",", " ").split():
                    try:
                        idx = int(num.strip()) - 1
                        if 0 <= idx < len(state.ideas):
                            idea_id = state.ideas[idx].id
                            current_ideas.append(idea_id)
                            state.ideas[idx].cluster_id = idea_id
                    except ValueError:
                        continue

        # Save last cluster
        if current_name:
            cluster = IdeaCluster(name=current_name, description=current_desc)
            cluster.idea_ids = current_ideas
            clusters.append(cluster)

        return clusters

    def _parse_scores(self, response: str, state: IdeationState) -> None:
        """Parse scores from response and update ideas.

        Args:
            response: Provider response
            state: Ideation state
        """
        for line in response.split("\n"):
            line = line.strip()
            if ":" in line:
                try:
                    parts = line.split(":")
                    num = int(parts[0].strip().rstrip("."))
                    score_part = parts[1].strip()
                    # Extract score (handle "0.8 - justification" format)
                    score_str = score_part.split()[0].split("-")[0].strip()
                    score = float(score_str)
                    if 0 <= score <= 1 and 0 < num <= len(state.ideas):
                        state.ideas[num - 1].score = score
                except (ValueError, IndexError):
                    continue

    def get_ideation(self, ideation_id: str) -> Optional[dict[str, Any]]:
        """Get full ideation details.

        Args:
            ideation_id: Ideation identifier

        Returns:
            Ideation data or None if not found
        """
        state = self.memory.load_ideation(ideation_id)
        if not state:
            return None

        return {
            "id": state.id,
            "topic": state.topic,
            "phase": state.phase.value,
            "perspectives": state.perspectives,
            "scoring_criteria": state.scoring_criteria,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "ideas": [
                {
                    "id": i.id,
                    "content": i.content,
                    "perspective": i.perspective,
                    "score": i.score,
                    "cluster_id": i.cluster_id,
                }
                for i in state.ideas
            ],
            "clusters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "idea_count": len(c.idea_ids),
                    "average_score": c.average_score,
                    "selected": c.selected_for_elaboration,
                    "has_elaboration": c.elaboration is not None,
                }
                for c in state.clusters
            ],
        }

    def list_ideations(self, limit: Optional[int] = 50) -> list[dict[str, Any]]:
        """List ideation sessions.

        Args:
            limit: Maximum sessions to return

        Returns:
            List of ideation summaries
        """
        ideations = self.memory.list_ideations(limit=limit)

        return [
            {
                "id": i.id,
                "topic": i.topic,
                "phase": i.phase.value,
                "idea_count": len(i.ideas),
                "cluster_count": len(i.clusters),
                "created_at": i.created_at.isoformat(),
                "updated_at": i.updated_at.isoformat(),
            }
            for i in ideations
        ]

    def delete_ideation(self, ideation_id: str) -> bool:
        """Delete an ideation session.

        Args:
            ideation_id: Ideation identifier

        Returns:
            True if deleted, False if not found
        """
        return self.memory.delete_ideation(ideation_id)
