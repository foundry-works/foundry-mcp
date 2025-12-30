"""THINKDEEP workflow for hypothesis-driven systematic investigation.

Provides deep investigation capabilities with hypothesis tracking,
evidence accumulation, and confidence progression.
"""

import logging
from typing import Any, Optional

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    Hypothesis,
    InvestigationStep,
    ThinkDeepState,
)
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult

logger = logging.getLogger(__name__)


class ThinkDeepWorkflow(ResearchWorkflowBase):
    """Hypothesis-driven systematic investigation workflow.

    Features:
    - Multi-step investigation with depth tracking
    - Hypothesis creation and tracking
    - Evidence accumulation (supporting/contradicting)
    - Confidence level progression
    - Convergence detection
    - State persistence across sessions
    """

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
    ) -> None:
        """Initialize thinkdeep workflow.

        Args:
            config: Research configuration
            memory: Optional memory instance
        """
        super().__init__(config, memory)

    def execute(
        self,
        topic: Optional[str] = None,
        investigation_id: Optional[str] = None,
        query: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute an investigation step.

        Either starts a new investigation (requires topic) or continues
        an existing one (requires investigation_id and query).

        Args:
            topic: Topic for new investigation
            investigation_id: Existing investigation to continue
            query: Follow-up query for continuing investigation
            system_prompt: System prompt for new investigations
            provider_id: Provider to use
            max_depth: Maximum investigation depth (uses config default if None)

        Returns:
            WorkflowResult with investigation findings
        """
        # Determine if starting new or continuing
        if investigation_id:
            state = self.memory.load_investigation(investigation_id)
            if not state:
                return WorkflowResult(
                    success=False,
                    content="",
                    error=f"Investigation {investigation_id} not found",
                )
            # Use query if provided, otherwise generate next question
            current_query = query or self._generate_next_query(state)
        elif topic:
            state = ThinkDeepState(
                topic=topic,
                max_depth=max_depth or self.config.thinkdeep_max_depth,
                system_prompt=system_prompt,
            )
            current_query = self._generate_initial_query(topic)
        else:
            return WorkflowResult(
                success=False,
                content="",
                error="Either 'topic' (for new investigation) or 'investigation_id' (to continue) is required",
            )

        # Check if already converged
        if state.converged:
            return WorkflowResult(
                success=True,
                content=self._format_summary(state),
                metadata={
                    "investigation_id": state.id,
                    "converged": True,
                    "convergence_reason": state.convergence_reason,
                    "hypothesis_count": len(state.hypotheses),
                    "step_count": len(state.steps),
                },
            )

        # Execute investigation step
        result = self._execute_investigation_step(
            state=state,
            query=current_query,
            provider_id=provider_id,
        )

        if not result.success:
            return result

        # Check for convergence
        state.check_convergence()

        # Persist state
        self.memory.save_investigation(state)

        # Add metadata
        result.metadata["investigation_id"] = state.id
        result.metadata["current_depth"] = state.current_depth
        result.metadata["max_depth"] = state.max_depth
        result.metadata["converged"] = state.converged
        result.metadata["hypothesis_count"] = len(state.hypotheses)
        result.metadata["step_count"] = len(state.steps)

        if state.converged:
            result.metadata["convergence_reason"] = state.convergence_reason

        return result

    def _generate_initial_query(self, topic: str) -> str:
        """Generate the initial investigation query.

        Args:
            topic: Investigation topic

        Returns:
            Initial query string
        """
        return f"Let's investigate: {topic}\n\nWhat are the key aspects we should explore? Please identify 2-3 initial hypotheses we can investigate."

    def _generate_next_query(self, state: ThinkDeepState) -> str:
        """Generate the next investigation query based on current state.

        Args:
            state: Current investigation state

        Returns:
            Next query string
        """
        # Summarize current hypotheses
        hyp_summary = "\n".join(
            f"- {h.statement} (confidence: {h.confidence.value})"
            for h in state.hypotheses
        )

        return f"""Based on our investigation so far:

Topic: {state.topic}

Current hypotheses:
{hyp_summary}

What additional evidence or questions should we explore to increase confidence in or refute these hypotheses?"""

    def _execute_investigation_step(
        self,
        state: ThinkDeepState,
        query: str,
        provider_id: Optional[str],
    ) -> WorkflowResult:
        """Execute a single investigation step.

        Args:
            state: Investigation state
            query: Query for this step
            provider_id: Provider to use

        Returns:
            WorkflowResult with step findings
        """
        # Build system prompt for investigation
        system_prompt = state.system_prompt or self._build_investigation_system_prompt()

        # Execute provider
        result = self._execute_provider(
            prompt=query,
            provider_id=provider_id,
            system_prompt=system_prompt,
        )

        if not result.success:
            return result

        # Create investigation step
        step = state.add_step(query=query, depth=state.current_depth)
        step.response = result.content
        step.provider_id = result.provider_id
        step.model_used = result.model_used

        # Parse and update hypotheses from response
        self._update_hypotheses_from_response(state, step, result.content)

        # Increment depth
        state.current_depth += 1

        return result

    def _build_investigation_system_prompt(self) -> str:
        """Build the system prompt for investigation.

        Returns:
            System prompt string
        """
        return """You are a systematic researcher conducting a deep investigation.

When analyzing topics:
1. Identify key hypotheses that could explain the phenomenon
2. Look for evidence that supports or contradicts each hypothesis
3. Update confidence levels based on evidence strength
4. Suggest next questions to increase understanding

For each response, structure your findings as:
- Key insights discovered
- Evidence for/against existing hypotheses
- New hypotheses to consider
- Recommended next steps

Be thorough but concise. Focus on advancing understanding systematically."""

    def _update_hypotheses_from_response(
        self,
        state: ThinkDeepState,
        step: InvestigationStep,
        response: str,
    ) -> None:
        """Parse response and update hypotheses.

        This is a simplified implementation that looks for hypothesis-related
        keywords. A more sophisticated version could use structured output
        or NLP to extract hypotheses more accurately.

        Args:
            state: Investigation state
            step: Current investigation step
            response: Provider response
        """
        response_lower = response.lower()

        # Simple heuristic: if this is early in investigation, look for new hypotheses
        if state.current_depth < 2:
            # Extract potential hypotheses (simplified)
            if "hypothesis" in response_lower or "suggests that" in response_lower:
                # For now, create a generic hypothesis if none exist
                if not state.hypotheses:
                    hyp = state.add_hypothesis(
                        statement=f"Initial investigation of: {state.topic}",
                        confidence=ConfidenceLevel.SPECULATION,
                    )
                    step.hypotheses_generated.append(hyp.id)

        # Update existing hypotheses based on evidence language
        for hyp in state.hypotheses:
            # Look for supporting evidence
            if any(
                phrase in response_lower
                for phrase in ["supports", "confirms", "evidence for", "consistent with"]
            ):
                hyp.add_evidence(f"Step {step.id}: {response[:200]}...", supporting=True)
                step.hypotheses_updated.append(hyp.id)

                # Update confidence if strong support
                if hyp.confidence == ConfidenceLevel.SPECULATION:
                    hyp.update_confidence(ConfidenceLevel.LOW)
                elif hyp.confidence == ConfidenceLevel.LOW:
                    hyp.update_confidence(ConfidenceLevel.MEDIUM)

            # Look for contradicting evidence
            if any(
                phrase in response_lower
                for phrase in ["contradicts", "refutes", "evidence against", "inconsistent"]
            ):
                hyp.add_evidence(f"Step {step.id}: {response[:200]}...", supporting=False)
                step.hypotheses_updated.append(hyp.id)

    def _format_summary(self, state: ThinkDeepState) -> str:
        """Format investigation summary.

        Args:
            state: Investigation state

        Returns:
            Formatted summary string
        """
        parts = [f"# Investigation Summary: {state.topic}\n"]

        if state.converged:
            parts.append(f"**Status**: Converged ({state.convergence_reason})\n")
        else:
            parts.append(f"**Status**: In progress (depth {state.current_depth}/{state.max_depth})\n")

        parts.append(f"**Steps completed**: {len(state.steps)}\n")
        parts.append(f"**Hypotheses tracked**: {len(state.hypotheses)}\n")

        if state.hypotheses:
            parts.append("\n## Hypotheses\n")
            for hyp in state.hypotheses:
                parts.append(f"### {hyp.statement}")
                parts.append(f"- Confidence: {hyp.confidence.value}")
                parts.append(f"- Supporting evidence: {len(hyp.supporting_evidence)}")
                parts.append(f"- Contradicting evidence: {len(hyp.contradicting_evidence)}\n")

        return "\n".join(parts)

    def get_investigation(self, investigation_id: str) -> Optional[dict[str, Any]]:
        """Get full investigation details.

        Args:
            investigation_id: Investigation identifier

        Returns:
            Investigation data or None if not found
        """
        state = self.memory.load_investigation(investigation_id)
        if not state:
            return None

        return {
            "id": state.id,
            "topic": state.topic,
            "current_depth": state.current_depth,
            "max_depth": state.max_depth,
            "converged": state.converged,
            "convergence_reason": state.convergence_reason,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "confidence": h.confidence.value,
                    "supporting_evidence_count": len(h.supporting_evidence),
                    "contradicting_evidence_count": len(h.contradicting_evidence),
                }
                for h in state.hypotheses
            ],
            "steps": [
                {
                    "id": s.id,
                    "depth": s.depth,
                    "query": s.query,
                    "response_preview": s.response[:200] + "..." if s.response and len(s.response) > 200 else s.response,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in state.steps
            ],
        }

    def list_investigations(self, limit: Optional[int] = 50) -> list[dict[str, Any]]:
        """List investigations.

        Args:
            limit: Maximum investigations to return

        Returns:
            List of investigation summaries
        """
        investigations = self.memory.list_investigations(limit=limit)

        return [
            {
                "id": i.id,
                "topic": i.topic,
                "current_depth": i.current_depth,
                "max_depth": i.max_depth,
                "converged": i.converged,
                "hypothesis_count": len(i.hypotheses),
                "step_count": len(i.steps),
                "created_at": i.created_at.isoformat(),
                "updated_at": i.updated_at.isoformat(),
            }
            for i in investigations
        ]

    def delete_investigation(self, investigation_id: str) -> bool:
        """Delete an investigation.

        Args:
            investigation_id: Investigation identifier

        Returns:
            True if deleted, False if not found
        """
        return self.memory.delete_investigation(investigation_id)
