"""Multi-agent supervisor orchestration for deep research.

Contains agent roles, decision tracking, supervisor hooks for workflow
event injection, and the orchestrator that coordinates phase transitions.
Includes optional LLM-driven reflection at phase boundaries.
"""

from __future__ import annotations

import json
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
from foundry_mcp.core.research.models.sources import SourceQuality

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Specialist agent roles in the multi-agent research workflow.

    Agent Responsibilities:
    - SUPERVISOR: Orchestrates phase transitions, evaluates quality gates,
      decides on iteration vs completion.
    - CLARIFIER: Evaluates query specificity and generates clarifying
      questions. Infers constraints from vague queries to focus research.
    - BRIEFER: Generates the enriched research brief from the raw query.
    - GATHERER: Executes parallel search across providers, handles rate
      limiting, deduplicates sources, and validates source quality.
    - SYNTHESIZER: Generates coherent report sections, ensures logical
      flow, integrates findings, and produces the final synthesis.
    """

    SUPERVISOR = "supervisor"
    CLARIFIER = "clarifier"
    BRIEFER = "briefer"
    GATHERER = "gatherer"
    SYNTHESIZER = "synthesizer"


# Mapping from workflow phases to specialist agents
PHASE_TO_AGENT: dict[DeepResearchPhase, AgentRole] = {
    DeepResearchPhase.CLARIFICATION: AgentRole.CLARIFIER,
    DeepResearchPhase.BRIEF: AgentRole.BRIEFER,
    DeepResearchPhase.GATHERING: AgentRole.GATHERER,
    DeepResearchPhase.SUPERVISION: AgentRole.SUPERVISOR,
    DeepResearchPhase.SYNTHESIS: AgentRole.SYNTHESIZER,
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


@dataclass
class ReflectionDecision:
    """Result of an LLM-driven reflection at a phase boundary.

    Captures the LLM's quality assessment and proceed/adjust recommendation
    so the supervisor can make informed decisions about workflow continuation.
    """

    quality_assessment: str  # LLM's assessment of phase output quality
    proceed: bool  # Whether to proceed to the next phase
    adjustments: list[str] = dataclass_field(default_factory=list)  # Suggested adjustments
    rationale: str = ""  # Why the LLM made this recommendation
    phase: str = ""  # Phase that was evaluated
    provider_id: Optional[str] = None  # Provider used for reflection
    model_used: Optional[str] = None  # Model used for reflection
    tokens_used: int = 0  # Tokens consumed by reflection call
    duration_ms: float = 0.0  # Duration of reflection call

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "quality_assessment": self.quality_assessment,
            "proceed": self.proceed,
            "adjustments": self.adjustments,
            "rationale": self.rationale,
            "phase": self.phase,
            "provider_id": self.provider_id,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
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
        - CLARIFICATION: system prompt
        - BRIEF: system prompt, constraints
        - GATHERING: sub-queries, source types, rate limits
        - SUPERVISION: completed queries, source count
        - SYNTHESIS: findings, gaps, research brief
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
        elif phase == DeepResearchPhase.BRIEF:
            return {
                **base_inputs,
                "system_prompt": state.system_prompt,
                "clarification_constraints": state.clarification_constraints,
            }
        elif phase == DeepResearchPhase.GATHERING:
            return {
                **base_inputs,
                "sub_queries": [q.query for q in state.pending_sub_queries()],
                "source_types": [st.value for st in state.source_types],
                "max_sources_per_query": state.max_sources_per_query,
            }
        elif phase == DeepResearchPhase.SUPERVISION:
            return {
                **base_inputs,
                "completed_sub_queries": len(state.completed_sub_queries()),
                "total_sources": len(state.sources),
                "supervision_round": state.supervision_round,
                "max_supervision_rounds": state.max_supervision_rounds,
            }
        elif phase == DeepResearchPhase.SYNTHESIS:
            return {
                **base_inputs,
                "finding_count": len(state.findings),
                "gap_count": len(state.gaps),
                "has_research_brief": state.research_brief is not None,
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
                "quality_ok": True,
                "rationale": (
                    f"Clarification {'provided constraints' if has_constraints else 'skipped/no constraints needed'}. "
                    "Proceeding to brief."
                ),
            }

        elif phase == DeepResearchPhase.BRIEF:
            has_brief = state.research_brief is not None
            return {
                "has_research_brief": has_brief,
                "quality_ok": has_brief,
                "rationale": (
                    f"Brief {'generated' if has_brief else 'missing'}. "
                    f"{'Proceeding to supervision' if has_brief else 'May need retry'}."
                ),
            }

        elif phase == DeepResearchPhase.GATHERING:
            source_count = len(state.sources)
            quality_ok = source_count >= 3
            return {
                "source_count": source_count,
                "quality_ok": quality_ok,
                "rationale": (
                    f"Gathering collected {source_count} sources. "
                    f"{'Sufficient' if quality_ok else 'May need more sources'}."
                ),
            }

        elif phase == DeepResearchPhase.SUPERVISION:
            pending = len(state.pending_sub_queries())
            return {
                "supervision_round": state.supervision_round,
                "pending_follow_ups": pending,
                "quality_ok": True,
                "rationale": f"Supervision round {state.supervision_round}: {pending} follow-up queries queued.",
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
                    f"{'Complete' if quality_ok else 'May need improvement'}."
                ),
            }

        return {"rationale": f"Phase {phase.value} completed", "quality_ok": True}

    def decide_iteration(self, state: DeepResearchState) -> AgentDecision:
        """Supervisor decides whether to iterate or complete.

        With the collapsed pipeline (no refinement phase), the supervisor
        always completes after synthesis. Supervision gap-filling handles
        iterative improvement before synthesis.

        Args:
            state: Current research state

        Returns:
            AgentDecision with complete decision
        """
        decision = AgentDecision(
            agent=AgentRole.SUPERVISOR,
            action="decide_iteration",
            rationale=f"Completing: iteration {state.iteration}/{state.max_iterations}",
            inputs={
                "iteration": state.iteration,
                "max_iterations": state.max_iterations,
            },
            outputs={
                "should_iterate": False,
                "next_phase": "COMPLETED",
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

    async def async_think_pause(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
        *,
        workflow: Any = None,
    ) -> ReflectionDecision:
        """Execute LLM-driven reflection at a phase boundary.

        Sends the phase results and state summary to a fast model, which
        assesses quality and recommends whether to proceed or adjust.

        Args:
            state: Current research state (after phase execution)
            phase: The phase that just completed
            workflow: The DeepResearchWorkflow instance (provides config, _execute_provider_async)

        Returns:
            ReflectionDecision with LLM assessment
        """
        if workflow is None:
            logger.warning("async_think_pause called without workflow instance, returning proceed=True")
            return ReflectionDecision(
                quality_assessment="No workflow context available",
                proceed=True,
                rationale="Skipped reflection: no workflow instance provided",
                phase=phase.value,
            )

        import time

        reflection_prompt = self._build_reflection_llm_prompt(state, phase)
        system_prompt = self._build_reflection_system_prompt()

        provider_id = workflow.config.get_reflection_provider()
        timeout = workflow.config.deep_research_reflection_timeout

        start_time = time.perf_counter()

        try:
            result = await workflow._execute_provider_async(
                prompt=reflection_prompt,
                provider_id=provider_id,
                model=None,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.2,  # Low temperature for analytical assessment
                phase="reflection",
                fallback_providers=[],
                max_retries=1,
                retry_delay=2.0,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Reflection LLM call failed for phase %s: %s. Proceeding with heuristic fallback.",
                phase.value,
                exc,
            )
            return ReflectionDecision(
                quality_assessment="Reflection call failed",
                proceed=True,
                rationale=f"LLM reflection error: {exc}. Falling back to heuristic.",
                phase=phase.value,
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if not result.success:
            logger.warning(
                "Reflection LLM returned failure for phase %s: %s. Using heuristic fallback.",
                phase.value,
                result.error,
            )
            return ReflectionDecision(
                quality_assessment="Reflection call returned failure",
                proceed=True,
                rationale=f"LLM reflection failed: {result.error}. Falling back to heuristic.",
                phase=phase.value,
                provider_id=result.provider_id,
                model_used=result.model_used,
                duration_ms=duration_ms,
            )

        decision = self._parse_reflection_response(
            result.content,
            phase=phase,
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used or 0,
            duration_ms=duration_ms,
        )

        # Record the reflection as an agent decision for traceability
        self._decisions.append(
            AgentDecision(
                agent=AgentRole.SUPERVISOR,
                action=f"reflect_{phase.value}",
                rationale=decision.rationale,
                inputs={"phase": phase.value, "reflection_prompt_length": len(reflection_prompt)},
                outputs=decision.to_dict(),
            )
        )

        return decision

    def _build_reflection_system_prompt(self) -> str:
        """Build system prompt for LLM reflection calls."""
        return """You are a research quality supervisor. Your task is to evaluate the quality of a completed research phase and recommend whether to proceed.

Respond with valid JSON in this exact structure:
{
    "quality_assessment": "Brief assessment of the phase output quality",
    "proceed": true/false,
    "adjustments": ["Optional suggestion 1", "Optional suggestion 2"],
    "rationale": "Why you recommend proceeding or not"
}

Rules:
- Set "proceed" to true if the phase produced usable output, even if imperfect
- Set "proceed" to false only if the output is fundamentally insufficient
- Keep adjustments practical and actionable (max 3)
- Be pragmatic: minor quality issues should not block progress
- Consider the research phase context when evaluating quality

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_reflection_llm_prompt(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> str:
        """Build the user prompt for LLM reflection, summarizing phase output.

        Args:
            state: Current research state
            phase: Phase that just completed

        Returns:
            Reflection prompt string with phase-specific context
        """
        base = (
            f"Research query: {state.original_query}\n"
            f"Phase just completed: {phase.value}\n"
            f"Iteration: {state.iteration}/{state.max_iterations}\n"
        )

        if phase == DeepResearchPhase.CLARIFICATION:
            has_constraints = bool(state.clarification_constraints)
            base += (
                f"\nConstraints inferred: {has_constraints}\n"
                f"Constraint keys: {list(state.clarification_constraints.keys()) if has_constraints else '(none)'}\n"
            )

        elif phase == DeepResearchPhase.BRIEF:
            base += (
                f"\nResearch brief available: {state.research_brief is not None}\n"
                f"Sub-queries generated: {len(state.sub_queries)}\n"
            )

        elif phase == DeepResearchPhase.GATHERING:
            base += (
                f"\nSources collected: {len(state.sources)}\n"
                f"Source quality distribution: "
                f"HIGH={len([s for s in state.sources if s.quality == SourceQuality.HIGH])}, "
                f"MEDIUM={len([s for s in state.sources if s.quality == SourceQuality.MEDIUM])}, "
                f"LOW={len([s for s in state.sources if s.quality == SourceQuality.LOW])}\n"
            )

        elif phase == DeepResearchPhase.SUPERVISION:
            pending = len(state.pending_sub_queries())
            completed = len(state.completed_sub_queries())
            base += (
                f"\nSupervision round: {state.supervision_round}/{state.max_supervision_rounds}\n"
                f"Completed sub-queries: {completed}\n"
                f"Pending follow-up queries: {pending}\n"
                f"Total sources: {len(state.sources)}\n"
            )

        elif phase == DeepResearchPhase.SYNTHESIS:
            report_length = len(state.report) if state.report else 0
            base += (
                f"\nReport generated: {state.report is not None}\n"
                f"Report length: {report_length} chars\n"
                f"Unresolved gaps: {len(state.unresolved_gaps())}\n"
            )

        base += "\nEvaluate: Is the output quality sufficient to proceed to the next phase?"
        return base

    def _parse_reflection_response(
        self,
        content: str,
        *,
        phase: DeepResearchPhase,
        provider_id: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
    ) -> ReflectionDecision:
        """Parse LLM reflection response into a ReflectionDecision.

        Falls back to proceed=True on parse failures.

        Args:
            content: Raw LLM response
            phase: Phase that was evaluated
            provider_id: Provider used
            model_used: Model used
            tokens_used: Tokens consumed
            duration_ms: Call duration

        Returns:
            ReflectionDecision
        """
        from foundry_mcp.core.research.workflows.deep_research._helpers import extract_json

        default = ReflectionDecision(
            quality_assessment="Unable to parse reflection response",
            proceed=True,
            rationale="Defaulting to proceed due to parse failure",
            phase=phase.value,
            provider_id=provider_id,
            model_used=model_used,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
        )

        if not content:
            return default

        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in reflection response for phase %s", phase.value)
            return default

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse reflection JSON for phase %s: %s", phase.value, e)
            return default

        adjustments_raw = data.get("adjustments", [])
        adjustments = [str(a) for a in adjustments_raw[:3] if a] if isinstance(adjustments_raw, list) else []

        return ReflectionDecision(
            quality_assessment=str(data.get("quality_assessment", "")),
            proceed=bool(data.get("proceed", True)),
            adjustments=adjustments,
            rationale=str(data.get("rationale", "")),
            phase=phase.value,
            provider_id=provider_id,
            model_used=model_used,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
        )

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
            DeepResearchPhase.BRIEF: (
                f"Brief generation complete. Research brief: {bool(state.research_brief)}. "
                "Evaluate: Is the brief comprehensive enough for supervision?"
            ),
            DeepResearchPhase.GATHERING: (
                f"Gathering complete. Collected {len(state.sources)} sources. "
                f"Evaluate: Is source diversity sufficient? Quality distribution?"
            ),
            DeepResearchPhase.SUPERVISION: (
                f"Supervision round {state.supervision_round} complete. "
                f"{len(state.pending_sub_queries())} follow-up queries pending. "
                "Evaluate: Is coverage sufficient or should gathering continue?"
            ),
            DeepResearchPhase.SYNTHESIS: (
                f"Synthesis complete. Report: {len(state.report or '')} chars. "
                f"Iteration {state.iteration}/{state.max_iterations}. "
                "Evaluate: Report quality?"
            ),
        }
        return prompts.get(phase, f"Phase {phase.value} complete. Evaluate progress.")
