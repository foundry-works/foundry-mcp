"""Multi-agent supervisor orchestration for deep research.

Contains agent roles, decision tracking, supervisor hooks for workflow
event injection, and the orchestrator that coordinates phase transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import SourceQuality

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Specialist agent roles in the multi-agent research workflow.

    Agent Responsibilities:
    - SUPERVISOR: Orchestrates phase transitions, evaluates quality gates,
      decides on iteration vs completion. The supervisor runs think-tool
      pauses between phases to evaluate progress and adjust strategy.
    - CLARIFIER: Evaluates query specificity and generates clarifying
      questions. Infers constraints from vague queries to focus research.
    - PLANNER: Decomposes the original query into focused sub-queries,
      generates the research brief, and identifies key themes to explore.
    - GATHERER: Executes parallel search across providers, handles rate
      limiting, deduplicates sources, and validates source quality.
    - ANALYZER: Extracts findings from sources, assesses evidence quality,
      identifies contradictions, and rates source reliability.
    - SYNTHESIZER: Generates coherent report sections, ensures logical
      flow, integrates findings, and produces the final synthesis.
    - REFINER: Identifies knowledge gaps, generates follow-up queries,
      determines if additional iteration is needed, and prioritizes gaps.
    """

    SUPERVISOR = "supervisor"
    CLARIFIER = "clarifier"
    PLANNER = "planner"
    GATHERER = "gatherer"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    REFINER = "refiner"


# Mapping from workflow phases to specialist agents
PHASE_TO_AGENT: dict[DeepResearchPhase, AgentRole] = {
    DeepResearchPhase.CLARIFICATION: AgentRole.CLARIFIER,
    DeepResearchPhase.PLANNING: AgentRole.PLANNER,
    DeepResearchPhase.GATHERING: AgentRole.GATHERER,
    DeepResearchPhase.ANALYSIS: AgentRole.ANALYZER,
    DeepResearchPhase.SYNTHESIS: AgentRole.SYNTHESIZER,
    DeepResearchPhase.REFINEMENT: AgentRole.REFINER,
}


@dataclass
class AgentDecision:
    """Records a decision made by an agent during workflow execution.

    Used for traceability and debugging. Each decision captures:
    - Which agent made the decision
    - What action was taken
    - The rationale behind the decision
    - Inputs provided to the agent
    - Outputs produced (if any)
    - Timestamp for ordering

    Handoff Protocol:
    - Inputs: The context passed to the agent (query, state summary, etc.)
    - Outputs: The results produced (sub-queries, findings, report sections)
    - The supervisor evaluates outputs before proceeding to next phase
    """

    agent: AgentRole
    action: str  # e.g., "decompose_query", "evaluate_phase", "decide_iteration"
    rationale: str  # Why this decision was made
    inputs: dict[str, Any]  # Context provided to the agent
    outputs: Optional[dict[str, Any]] = None  # Results produced
    timestamp: datetime = dataclass_field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent": self.agent.value,
            "action": self.action,
            "rationale": self.rationale,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "timestamp": self.timestamp.isoformat(),
        }


class SupervisorHooks:
    """Hooks for multi-agent supervisor orchestration.

    Allows external orchestrators to inject behavior at key workflow
    points, enabling think-tool pauses, agent handoffs, and custom
    routing logic.
    """

    def __init__(self) -> None:
        """Initialize with no-op defaults."""
        self._on_phase_start: Optional[Callable[[DeepResearchState], None]] = None
        self._on_phase_complete: Optional[Callable[[DeepResearchState], None]] = None
        self._on_think_pause: Optional[Callable[[DeepResearchState, str], str]] = None
        self._on_agent_handoff: Optional[Callable[[str, dict], dict]] = None

    def on_phase_start(self, callback: Callable[[DeepResearchState], None]) -> None:
        """Register callback for phase start events."""
        self._on_phase_start = callback

    def on_phase_complete(self, callback: Callable[[DeepResearchState], None]) -> None:
        """Register callback for phase completion events."""
        self._on_phase_complete = callback

    def on_think_pause(self, callback: Callable[[DeepResearchState, str], str]) -> None:
        """Register callback for think-tool pauses.

        The callback receives the current state and a reflection prompt,
        and should return guidance for the next step.
        """
        self._on_think_pause = callback

    def on_agent_handoff(self, callback: Callable[[str, dict], dict]) -> None:
        """Register callback for agent handoffs.

        The callback receives the target agent name and context dict,
        and should return the agent's response.
        """
        self._on_agent_handoff = callback

    def emit_phase_start(self, state: DeepResearchState) -> None:
        """Emit phase start event."""
        if self._on_phase_start:
            try:
                self._on_phase_start(state)
            except Exception as exc:
                logger.error("Phase start hook failed: %s", exc)

    def emit_phase_complete(self, state: DeepResearchState) -> None:
        """Emit phase complete event."""
        if self._on_phase_complete:
            try:
                self._on_phase_complete(state)
            except Exception as exc:
                logger.error("Phase complete hook failed: %s", exc)

    def think_pause(self, state: DeepResearchState, prompt: str) -> Optional[str]:
        """Execute think pause if callback registered."""
        if self._on_think_pause:
            try:
                return self._on_think_pause(state, prompt)
            except Exception as exc:
                logger.error("Think pause hook failed: %s", exc)
        return None

    def agent_handoff(self, agent: str, context: dict) -> Optional[dict]:
        """Execute agent handoff if callback registered."""
        if self._on_agent_handoff:
            try:
                return self._on_agent_handoff(agent, context)
            except Exception as exc:
                logger.error("Agent handoff hook failed: %s", exc)
        return None


class SupervisorOrchestrator:
    """Coordinates specialist agents and manages phase transitions.

    The supervisor is responsible for:
    1. Deciding which specialist agent to dispatch for each phase
    2. Evaluating phase completion quality before proceeding
    3. Inserting think-tool pauses for reflection and strategy adjustment
    4. Recording all decisions for traceability
    5. Managing iteration vs completion decisions

    The orchestrator integrates with SupervisorHooks to allow external
    customization of decision logic (e.g., via LLM-based evaluation).

    Phase Dispatch Flow:
    ```
    SUPERVISOR -> evaluate context -> dispatch to PLANNER
                                   -> think pause (evaluate planning quality)
                                   -> dispatch to GATHERER
                                   -> think pause (evaluate source quality)
                                   -> dispatch to ANALYZER
                                   -> think pause (evaluate findings)
                                   -> dispatch to SYNTHESIZER
                                   -> think pause (evaluate report)
                                   -> decide: complete OR dispatch to REFINER
    ```
    """

    def __init__(self) -> None:
        """Initialize the supervisor orchestrator."""
        self._decisions: list[AgentDecision] = []

    def dispatch_to_agent(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> AgentDecision:
        """Dispatch work to the appropriate specialist agent for a phase.

        Args:
            state: Current research state
            phase: The phase to execute

        Returns:
            AgentDecision recording the dispatch
        """
        agent = PHASE_TO_AGENT.get(phase, AgentRole.SUPERVISOR)
        inputs = self._build_agent_inputs(state, phase)

        decision = AgentDecision(
            agent=agent,
            action=f"execute_{phase.value}",
            rationale=f"Phase {phase.value} requires {agent.value} specialist",
            inputs=inputs,
        )

        self._decisions.append(decision)
        return decision

    def _build_agent_inputs(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> dict[str, Any]:
        """Build the input context for a specialist agent.

        Handoff inputs vary by phase:
        - PLANNING: original query, system prompt
        - GATHERING: sub-queries, source types, rate limits
        - ANALYSIS: sources, findings so far
        - SYNTHESIS: findings, gaps, iteration count
        - REFINEMENT: gaps, remaining iterations, report draft
        """
        base_inputs = {
            "research_id": state.id,
            "original_query": state.original_query,
            "current_phase": phase.value,
            "iteration": state.iteration,
        }

        if phase == DeepResearchPhase.CLARIFICATION:
            return {
                **base_inputs,
                "system_prompt": state.system_prompt,
            }
        elif phase == DeepResearchPhase.PLANNING:
            return {
                **base_inputs,
                "system_prompt": state.system_prompt,
                "max_sub_queries": state.max_sub_queries,
                "clarification_constraints": state.clarification_constraints,
            }
        elif phase == DeepResearchPhase.GATHERING:
            return {
                **base_inputs,
                "sub_queries": [q.query for q in state.pending_sub_queries()],
                "source_types": [st.value for st in state.source_types],
                "max_sources_per_query": state.max_sources_per_query,
            }
        elif phase == DeepResearchPhase.ANALYSIS:
            return {
                **base_inputs,
                "source_count": len(state.sources),
                "high_quality_sources": len([s for s in state.sources if s.quality == SourceQuality.HIGH]),
            }
        elif phase == DeepResearchPhase.SYNTHESIS:
            return {
                **base_inputs,
                "finding_count": len(state.findings),
                "gap_count": len(state.gaps),
                "has_research_brief": state.research_brief is not None,
            }
        elif phase == DeepResearchPhase.REFINEMENT:
            return {
                **base_inputs,
                "gaps": [g.description for g in state.gaps if not g.resolved],
                "remaining_iterations": state.max_iterations - state.iteration,
                "has_report_draft": state.report is not None,
            }
        return base_inputs

    def evaluate_phase_completion(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> AgentDecision:
        """Supervisor evaluates whether a phase completed successfully.

        This is the think-tool pause where the supervisor reflects on
        the phase's outputs and decides whether to proceed.

        Args:
            state: Current research state (after phase execution)
            phase: The phase that just completed

        Returns:
            AgentDecision with evaluation and proceed/retry rationale
        """
        evaluation = self._evaluate_phase_quality(state, phase)

        decision = AgentDecision(
            agent=AgentRole.SUPERVISOR,
            action="evaluate_phase",
            rationale=evaluation["rationale"],
            inputs={
                "phase": phase.value,
                "iteration": state.iteration,
            },
            outputs=evaluation,
        )

        self._decisions.append(decision)
        return decision

    def _evaluate_phase_quality(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> dict[str, Any]:
        """Evaluate the quality of a completed phase.

        Returns metrics and a proceed/retry recommendation.
        """
        if phase == DeepResearchPhase.CLARIFICATION:
            has_constraints = bool(state.clarification_constraints)
            return {
                "has_constraints": has_constraints,
                "quality_ok": True,  # Clarification always proceeds
                "rationale": (
                    f"Clarification {'provided constraints' if has_constraints else 'skipped/no constraints needed'}. "
                    "Proceeding to planning."
                ),
            }

        elif phase == DeepResearchPhase.PLANNING:
            sub_query_count = len(state.sub_queries)
            quality_ok = sub_query_count >= 2  # At least 2 sub-queries
            return {
                "sub_query_count": sub_query_count,
                "has_research_brief": state.research_brief is not None,
                "quality_ok": quality_ok,
                "rationale": (
                    f"Planning produced {sub_query_count} sub-queries. "
                    f"{'Sufficient' if quality_ok else 'Insufficient'} for gathering."
                ),
            }

        elif phase == DeepResearchPhase.GATHERING:
            source_count = len(state.sources)
            quality_ok = source_count >= 3  # At least 3 sources
            return {
                "source_count": source_count,
                "quality_ok": quality_ok,
                "rationale": (
                    f"Gathering collected {source_count} sources. "
                    f"{'Sufficient' if quality_ok else 'May need more sources'}."
                ),
            }

        elif phase == DeepResearchPhase.ANALYSIS:
            finding_count = len(state.findings)
            high_confidence = len([f for f in state.findings if f.confidence == ConfidenceLevel.HIGH])
            quality_ok = finding_count >= 2
            return {
                "finding_count": finding_count,
                "high_confidence_count": high_confidence,
                "quality_ok": quality_ok,
                "rationale": (
                    f"Analysis extracted {finding_count} findings "
                    f"({high_confidence} high confidence). "
                    f"{'Ready for synthesis' if quality_ok else 'May need more analysis'}."
                ),
            }

        elif phase == DeepResearchPhase.SYNTHESIS:
            has_report = state.report is not None
            report_length = len(state.report) if state.report else 0
            quality_ok = has_report and report_length > 100
            return {
                "has_report": has_report,
                "report_length": report_length,
                "quality_ok": quality_ok,
                "rationale": (
                    f"Synthesis {'produced' if has_report else 'failed to produce'} report "
                    f"({report_length} chars). "
                    f"{'Complete' if quality_ok else 'May need refinement'}."
                ),
            }

        elif phase == DeepResearchPhase.REFINEMENT:
            unaddressed_gaps = len([g for g in state.gaps if not g.resolved])
            can_iterate = state.iteration < state.max_iterations
            should_iterate = unaddressed_gaps > 0 and can_iterate
            return {
                "unaddressed_gaps": unaddressed_gaps,
                "iteration": state.iteration,
                "max_iterations": state.max_iterations,
                "should_iterate": should_iterate,
                "rationale": (
                    f"Refinement found {unaddressed_gaps} gaps. "
                    f"{'Will iterate' if should_iterate else 'Completing'} "
                    f"(iteration {state.iteration}/{state.max_iterations})."
                ),
            }

        return {"rationale": f"Phase {phase.value} completed", "quality_ok": True}

    def decide_iteration(self, state: DeepResearchState) -> AgentDecision:
        """Supervisor decides whether to iterate or complete.

        Called after synthesis to determine if refinement is needed.

        Args:
            state: Current research state

        Returns:
            AgentDecision with iterate vs complete decision
        """
        unaddressed_gaps = [g for g in state.gaps if not g.resolved]
        can_iterate = state.iteration < state.max_iterations
        should_iterate = len(unaddressed_gaps) > 0 and can_iterate

        decision = AgentDecision(
            agent=AgentRole.SUPERVISOR,
            action="decide_iteration",
            rationale=(
                f"{'Iterating' if should_iterate else 'Completing'}: "
                f"{len(unaddressed_gaps)} gaps, "
                f"iteration {state.iteration}/{state.max_iterations}"
            ),
            inputs={
                "gap_count": len(unaddressed_gaps),
                "iteration": state.iteration,
                "max_iterations": state.max_iterations,
            },
            outputs={
                "should_iterate": should_iterate,
                "next_phase": (DeepResearchPhase.REFINEMENT.value if should_iterate else "COMPLETED"),
            },
        )

        self._decisions.append(decision)
        return decision

    def record_to_state(self, state: DeepResearchState) -> None:
        """Record all decisions to the state's metadata for persistence.

        Args:
            state: Research state to update
        """
        if "agent_decisions" not in state.metadata:
            state.metadata["agent_decisions"] = []

        state.metadata["agent_decisions"].extend([d.to_dict() for d in self._decisions])
        self._decisions.clear()

    def get_reflection_prompt(self, state: DeepResearchState, phase: DeepResearchPhase) -> str:
        """Generate a reflection prompt for the supervisor think pause.

        Args:
            state: Current research state
            phase: Phase that just completed

        Returns:
            Prompt for supervisor reflection
        """
        prompts = {
            DeepResearchPhase.CLARIFICATION: (
                f"Clarification complete. Constraints: {bool(state.clarification_constraints)}. "
                "Evaluate: Is the query now specific enough for focused research?"
            ),
            DeepResearchPhase.PLANNING: (
                f"Planning complete. Generated {len(state.sub_queries)} sub-queries. "
                f"Research brief: {bool(state.research_brief)}. "
                "Evaluate: Are sub-queries comprehensive? Any gaps in coverage?"
            ),
            DeepResearchPhase.GATHERING: (
                f"Gathering complete. Collected {len(state.sources)} sources. "
                f"Evaluate: Is source diversity sufficient? Quality distribution?"
            ),
            DeepResearchPhase.ANALYSIS: (
                f"Analysis complete. Extracted {len(state.findings)} findings, "
                f"identified {len(state.gaps)} gaps. "
                "Evaluate: Are findings well-supported? Critical gaps?"
            ),
            DeepResearchPhase.SYNTHESIS: (
                f"Synthesis complete. Report: {len(state.report or '')} chars. "
                f"Iteration {state.iteration}/{state.max_iterations}. "
                "Evaluate: Report quality? Need refinement?"
            ),
            DeepResearchPhase.REFINEMENT: (
                f"Refinement complete. Gaps addressed: "
                f"{len([g for g in state.gaps if g.resolved])}/{len(state.gaps)}. "
                "Evaluate: Continue iterating or finalize?"
            ),
        }
        return prompts.get(phase, f"Phase {phase.value} complete. Evaluate progress.")
