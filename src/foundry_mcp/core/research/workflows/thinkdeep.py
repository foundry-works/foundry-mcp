"""THINKDEEP workflow for hypothesis-driven systematic investigation.

Provides deep investigation capabilities with hypothesis tracking,
evidence accumulation, and confidence progression.
"""

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.thinkdeep import (
    InvestigationStep,
    ThinkDeepState,
)
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult

logger = logging.getLogger(__name__)


# --- Structured output models for LLM response parsing ---


class EvidenceItem(BaseModel):
    """A single piece of evidence from the LLM response."""

    text: str = Field(..., description="Description of the evidence")
    strength: Literal["strong", "moderate", "weak"] = Field(default="moderate", description="Strength of this evidence")
    supporting: bool = Field(default=True, description="True if supporting, False if contradicting")


class HypothesisUpdate(BaseModel):
    """An update to an existing hypothesis or a new hypothesis."""

    statement: str = Field(..., description="The hypothesis statement")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    is_new: bool = Field(default=False, description="True if this is a newly proposed hypothesis")


class ThinkDeepStructuredResponse(BaseModel):
    """Structured LLM response for ThinkDeep investigation steps."""

    hypotheses: list[HypothesisUpdate] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list, description="Suggested next investigation questions")
    key_insights: list[str] = Field(default_factory=list, description="Key insights from this step")


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
        try:
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
        except Exception as exc:
            logger.exception("ThinkDeepWorkflow.execute() failed with unexpected error: %s", exc)
            error_msg = str(exc) if str(exc) else exc.__class__.__name__
            return WorkflowResult(
                success=False,
                content="",
                error=f"ThinkDeep workflow failed: {error_msg}",
                metadata={
                    "workflow": "thinkdeep",
                    "error_type": exc.__class__.__name__,
                },
            )

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
        hyp_summary = "\n".join(f"- {h.statement} (confidence: {h.confidence.value})" for h in state.hypotheses)

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
        parse_method = self._update_hypotheses_from_response(state, step, result.content)

        # Increment depth
        state.current_depth += 1

        result.metadata["parse_method"] = parse_method
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

Your response MUST be valid JSON with this exact structure:
{
    "hypotheses": [
        {
            "statement": "The hypothesis statement",
            "evidence": [
                {
                    "text": "Description of the evidence",
                    "strength": "strong|moderate|weak",
                    "supporting": true
                }
            ],
            "is_new": true
        }
    ],
    "next_questions": ["Question to explore next"],
    "key_insights": ["Key insight discovered"]
}

Guidelines:
- For new hypotheses, set "is_new": true
- For existing hypotheses being updated with evidence, set "is_new": false and restate the hypothesis
- "supporting": true means evidence supports the hypothesis, false means it contradicts
- "strength": "strong" = highly conclusive, "moderate" = suggestive, "weak" = tangential
- Include 1-3 next questions to guide further investigation
- Include 1-3 key insights from this step

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _update_hypotheses_from_response(
        self,
        state: ThinkDeepState,
        step: InvestigationStep,
        response: str,
    ) -> str:
        """Parse response and update hypotheses.

        Attempts JSON structured output first, falls back to keyword matching.

        Args:
            state: Investigation state
            step: Current investigation step
            response: Provider response

        Returns:
            Parse method used: "json" or "fallback_keyword"
        """
        parsed = self._try_parse_structured_response(response)
        if parsed is not None:
            self._apply_structured_response(state, step, parsed)
            return "json"

        logger.warning("ThinkDeep: JSON parse failed, falling back to keyword extraction")
        self._apply_keyword_fallback(state, step, response)
        return "fallback_keyword"

    def _try_parse_structured_response(self, response: str) -> ThinkDeepStructuredResponse | None:
        """Attempt to parse a structured JSON response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed response or None if parsing fails
        """
        # Try to extract JSON from the response (may be wrapped in markdown)
        json_str = self._extract_json(response)
        if json_str is None:
            return None

        try:
            data = json.loads(json_str)
            return ThinkDeepStructuredResponse.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("ThinkDeep structured parse failed: %s", exc)
            return None

    @staticmethod
    def _extract_json(content: str) -> str | None:
        """Extract JSON object from content that may contain other text.

        Args:
            content: Raw content that may contain JSON

        Returns:
            Extracted JSON string or None
        """
        import re

        # Try code blocks first
        for match in re.findall(r"```(?:json)?\s*([\s\S]*?)```", content):
            match = match.strip()
            if match.startswith("{"):
                return match

        # Try raw JSON object
        brace_start = content.find("{")
        if brace_start == -1:
            return None

        depth = 0
        for i, char in enumerate(content[brace_start:], brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[brace_start : i + 1]
        return None

    def _apply_structured_response(
        self,
        state: ThinkDeepState,
        step: InvestigationStep,
        parsed: ThinkDeepStructuredResponse,
    ) -> None:
        """Apply a successfully parsed structured response to state.

        Args:
            state: Investigation state
            step: Current investigation step
            parsed: Parsed structured response
        """
        _STRENGTH_TO_CONFIDENCE: dict[str, ConfidenceLevel] = {
            "strong": ConfidenceLevel.MEDIUM,
            "moderate": ConfidenceLevel.LOW,
            "weak": ConfidenceLevel.SPECULATION,
        }

        for hyp_update in parsed.hypotheses:
            if hyp_update.is_new:
                # Create new hypothesis
                hyp = state.add_hypothesis(
                    statement=hyp_update.statement,
                    confidence=ConfidenceLevel.SPECULATION,
                )
                step.hypotheses_generated.append(hyp.id)

                # Apply evidence to new hypothesis
                for ev in hyp_update.evidence:
                    hyp.add_evidence(f"Step {step.id}: {ev.text}", supporting=ev.supporting)

                # Set confidence based on strongest supporting evidence
                supporting_evidence = [e for e in hyp_update.evidence if e.supporting]
                if supporting_evidence:
                    best_strength = min(
                        supporting_evidence,
                        key=lambda e: ["strong", "moderate", "weak"].index(e.strength),
                    ).strength
                    hyp.update_confidence(_STRENGTH_TO_CONFIDENCE.get(best_strength, ConfidenceLevel.SPECULATION))
            else:
                # Update existing hypothesis — match by statement similarity
                matched_hyp = self._find_matching_hypothesis(state, hyp_update.statement)
                if matched_hyp is None:
                    # No match found; treat as new
                    matched_hyp = state.add_hypothesis(
                        statement=hyp_update.statement,
                        confidence=ConfidenceLevel.SPECULATION,
                    )
                    step.hypotheses_generated.append(matched_hyp.id)

                for ev in hyp_update.evidence:
                    matched_hyp.add_evidence(f"Step {step.id}: {ev.text}", supporting=ev.supporting)
                    step.hypotheses_updated.append(matched_hyp.id)

                # Update confidence based on evidence strength
                supporting = [e for e in hyp_update.evidence if e.supporting]
                contradicting = [e for e in hyp_update.evidence if not e.supporting]
                if supporting and not contradicting:
                    best = min(supporting, key=lambda e: ["strong", "moderate", "weak"].index(e.strength)).strength
                    target = _STRENGTH_TO_CONFIDENCE.get(best, ConfidenceLevel.SPECULATION)
                    # Only increase confidence
                    confidence_order = list(ConfidenceLevel)
                    if confidence_order.index(target) > confidence_order.index(matched_hyp.confidence):
                        matched_hyp.update_confidence(target)

    @staticmethod
    def _find_matching_hypothesis(state: ThinkDeepState, statement: str) -> Any:
        """Find a hypothesis matching the given statement.

        Uses simple substring matching. Returns the first match or None.
        """
        statement_lower = statement.lower()
        for hyp in state.hypotheses:
            if hyp.statement.lower() in statement_lower or statement_lower in hyp.statement.lower():
                return hyp
        return None

    def _apply_keyword_fallback(
        self,
        state: ThinkDeepState,
        step: InvestigationStep,
        response: str,
    ) -> None:
        """Original keyword-based hypothesis extraction (fallback).

        Args:
            state: Investigation state
            step: Current investigation step
            response: Provider response
        """
        response_lower = response.lower()

        # Simple heuristic: if this is early in investigation, look for new hypotheses
        if state.current_depth < 2:
            if "hypothesis" in response_lower or "suggests that" in response_lower:
                if not state.hypotheses:
                    hyp = state.add_hypothesis(
                        statement=f"Initial investigation of: {state.topic}",
                        confidence=ConfidenceLevel.SPECULATION,
                    )
                    step.hypotheses_generated.append(hyp.id)

        # Update existing hypotheses based on evidence language
        for hyp in state.hypotheses:
            if any(phrase in response_lower for phrase in ["supports", "confirms", "evidence for", "consistent with"]):
                hyp.add_evidence(f"Step {step.id}: {response[:200]}...", supporting=True)
                step.hypotheses_updated.append(hyp.id)

                if hyp.confidence == ConfidenceLevel.SPECULATION:
                    hyp.update_confidence(ConfidenceLevel.LOW)
                elif hyp.confidence == ConfidenceLevel.LOW:
                    hyp.update_confidence(ConfidenceLevel.MEDIUM)

            if any(
                phrase in response_lower for phrase in ["contradicts", "refutes", "evidence against", "inconsistent"]
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
                    "response_preview": s.response[:200] + "..."
                    if s.response and len(s.response) > 200
                    else s.response,
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
