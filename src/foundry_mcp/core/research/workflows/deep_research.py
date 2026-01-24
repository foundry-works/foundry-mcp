"""Deep Research workflow with async background execution.

Provides multi-phase iterative research through query decomposition,
parallel source gathering, content analysis, and synthesized reporting.

Key Features:
- Background execution via daemon threads with asyncio.run()
- Immediate research_id return on start
- Status polling while running
- Task lifecycle tracking with cancellation support
- Multi-agent supervisor orchestration hooks

Note: Uses daemon threads (not asyncio.create_task()) to ensure background
execution works correctly from synchronous MCP tool handlers where there
is no running event loop.

Inspired by:
- open_deep_research: Multi-agent supervision with think-tool pauses
- Claude-Deep-Research: Dual-source search with link following
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4
from weakref import WeakValueDictionary

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    DeepResearchPhase,
    DeepResearchState,
    DOMAIN_TIERS,
    PhaseMetrics,
    ResearchMode,
    ResearchSource,
    SourceQuality,
)
from foundry_mcp.core.error_collection import ErrorRecord
from foundry_mcp.core.error_store import FileErrorStore
from foundry_mcp.core.providers import ContextWindowError
from foundry_mcp.core.research.providers import (
    SearchProvider,
    SearchProviderError,
    GoogleSearchProvider,
    PerplexitySearchProvider,
    SemanticScholarProvider,
    TavilySearchProvider,
)
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult

logger = logging.getLogger(__name__)


# =============================================================================
# Crash Handler Infrastructure
# =============================================================================

# Track active research sessions for crash recovery
_active_research_sessions: dict[str, "DeepResearchState"] = {}


def _crash_handler(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
    """Handle uncaught exceptions by logging to stderr and writing crash markers.

    This handler catches process-level crashes that escape normal exception handling
    and ensures we have visibility into what went wrong.
    """
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # Always write to stderr for visibility
    print(
        f"\n{'='*60}\n"
        f"DEEP RESEARCH CRASH HANDLER\n"
        f"{'='*60}\n"
        f"Exception: {exc_type.__name__}: {exc_value}\n"
        f"Active sessions: {list(_active_research_sessions.keys())}\n"
        f"Traceback:\n{tb_str}"
        f"{'='*60}\n",
        file=sys.stderr,
        flush=True,
    )

    # Try to save crash markers for active research sessions
    for research_id, state in _active_research_sessions.items():
        try:
            state.metadata["crash"] = True
            state.metadata["crash_error"] = str(exc_value)
            # Write crash marker file
            crash_path = (
                Path.home()
                / ".foundry-mcp"
                / "research"
                / "deep_research"
                / f"{research_id}.crash"
            )
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            crash_path.write_text(tb_str)
        except Exception:
            pass  # Best effort - don't fail the crash handler

    # Call original handler
    sys.__excepthook__(exc_type, exc_value, exc_tb)


# Install crash handler
sys.excepthook = _crash_handler


@atexit.register
def _cleanup_on_exit() -> None:
    """Mark any active sessions as interrupted on normal exit."""
    for research_id, state in _active_research_sessions.items():
        if state.completed_at is None:
            state.metadata["interrupted"] = True


# =============================================================================
# Domain-Based Source Quality Assessment
# =============================================================================


def _extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL.

    Args:
        url: Full URL string

    Returns:
        Domain string (e.g., "arxiv.org") or None if extraction fails
    """
    if not url:
        return None
    try:
        # Handle URLs without scheme
        if "://" not in url:
            url = "https://" + url
        # Extract domain using simple parsing
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else None
    except Exception:
        return None


def _extract_hostname(url: str) -> Optional[str]:
    """Extract full hostname from URL (preserves subdomains like www.).

    Args:
        url: Full URL string

    Returns:
        Full hostname (e.g., "www.arxiv.org", "docs.python.org") or None
    """
    if not url:
        return None
    try:
        # Handle URLs without scheme
        if "://" not in url:
            url = "https://" + url
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower() if parsed.netloc else None
    except Exception:
        return None


def _domain_matches_pattern(domain: str, pattern: str) -> bool:
    """Check if domain matches a pattern (supports wildcards).

    Patterns:
    - "arxiv.org" - exact match
    - "*.edu" - matches stanford.edu, mit.edu, etc.
    - "docs.*" - matches docs.python.org, docs.microsoft.com, etc.

    Args:
        domain: Domain to check (e.g., "stanford.edu")
        pattern: Pattern to match (e.g., "*.edu")

    Returns:
        True if domain matches pattern
    """
    pattern = pattern.lower()
    domain = domain.lower()

    if "*" not in pattern:
        # Exact match or subdomain match
        return domain == pattern or domain.endswith("." + pattern)

    if pattern.startswith("*."):
        # Suffix pattern: *.edu matches stanford.edu
        suffix = pattern[2:]
        return domain == suffix or domain.endswith("." + suffix)

    if pattern.endswith(".*"):
        # Prefix pattern: docs.* matches docs.python.org
        prefix = pattern[:-2]
        return domain == prefix or domain.startswith(prefix + ".")

    # General wildcard (treat as contains)
    return pattern.replace("*", "") in domain


def get_domain_quality(url: str, mode: ResearchMode) -> SourceQuality:
    """Determine source quality based on domain and research mode.

    Args:
        url: Source URL
        mode: Research mode (general, academic, technical)

    Returns:
        SourceQuality based on domain tier matching
    """
    domain = _extract_domain(url)
    if not domain:
        return SourceQuality.UNKNOWN

    tiers = DOMAIN_TIERS.get(mode.value, DOMAIN_TIERS["general"])

    # Check high-priority domains first
    for pattern in tiers.get("high", []):
        if _domain_matches_pattern(domain, pattern):
            return SourceQuality.HIGH

    # Check low-priority domains
    for pattern in tiers.get("low", []):
        if _domain_matches_pattern(domain, pattern):
            return SourceQuality.LOW

    # Default to medium for unmatched domains
    return SourceQuality.MEDIUM


def _normalize_title(title: str) -> str:
    """Normalize title for deduplication matching.

    Converts to lowercase, removes punctuation, and collapses whitespace
    to enable matching the same paper from different sources (e.g., arXiv vs OpenReview).

    Args:
        title: Source title to normalize

    Returns:
        Normalized title string for comparison
    """
    if not title:
        return ""
    # Lowercase, remove punctuation, collapse whitespace
    normalized = title.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# =============================================================================
# Task Lifecycle
# =============================================================================


class TaskStatus(str, Enum):
    """Status of a background research task."""

    PENDING = "pending"  # Created but not started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with error
    CANCELLED = "cancelled"  # Cancelled by user
    TIMEOUT = "timeout"  # Exceeded timeout limit


class AgentRole(str, Enum):
    """Specialist agent roles in the multi-agent research workflow.

    Agent Responsibilities:
    - SUPERVISOR: Orchestrates phase transitions, evaluates quality gates,
      decides on iteration vs completion. The supervisor runs think-tool
      pauses between phases to evaluate progress and adjust strategy.
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
    PLANNER = "planner"
    GATHERER = "gatherer"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"
    REFINER = "refiner"


# Mapping from workflow phases to specialist agents
PHASE_TO_AGENT: dict[DeepResearchPhase, AgentRole] = {
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
    timestamp: datetime = dataclass_field(default_factory=datetime.utcnow)

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


class BackgroundTask:
    """Tracks a background research task.

    Supports both asyncio.Task-based and thread-based execution for
    lifecycle management, cancellation support, and timeout handling.
    """

    def __init__(
        self,
        research_id: str,
        task: Optional[asyncio.Task[WorkflowResult]] = None,
        thread: Optional[threading.Thread] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize background task.

        Args:
            research_id: ID of the research session
            task: Optional asyncio task running the workflow
            thread: Optional thread running the workflow (preferred)
            timeout: Optional timeout in seconds
        """
        self.research_id = research_id
        self.task = task
        self.thread = thread
        self.timeout = timeout
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[WorkflowResult] = None
        # Event for signaling cancellation to thread-based execution
        self._cancel_event = threading.Event()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000

    @property
    def is_timed_out(self) -> bool:
        """Check if task has exceeded timeout."""
        if self.timeout is None:
            return False
        return (time.time() - self.started_at) > self.timeout

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()

    @property
    def is_done(self) -> bool:
        """Check if the task is done (for both thread and asyncio modes).

        Returns:
            True if the task has completed, False if still running.
        """
        if self.thread is not None:
            return not self.thread.is_alive()
        elif self.task is not None:
            return self.task.done()
        # Neither thread nor task - consider done (shouldn't happen)
        return True

    def cancel(self) -> bool:
        """Cancel the task.

        Returns:
            True if cancellation was requested, False if already done
        """
        # Handle thread-based execution
        if self.thread is not None:
            if not self.thread.is_alive():
                return False
            self._cancel_event.set()
            # Give thread a chance to clean up
            self.thread.join(timeout=5.0)
            self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()
            return True
        # Handle asyncio-based execution
        elif self.task is not None:
            if self.task.done():
                return False
            self.task.cancel()
            self.status = TaskStatus.CANCELLED
            self.completed_at = time.time()
            return True
        return False

    def mark_completed(self, result: WorkflowResult) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.result = result
        self.completed_at = time.time()
        if not result.success:
            self.error = result.error

    def mark_timeout(self) -> None:
        """Mark task as timed out."""
        self.status = TaskStatus.TIMEOUT
        self.completed_at = time.time()
        self.error = f"Task exceeded timeout of {self.timeout}s"
        # Signal cancellation
        if self.thread is not None:
            self._cancel_event.set()
        elif self.task is not None:
            self.task.cancel()


# =============================================================================
# Supervisor Hooks (Multi-Agent Orchestration)
# =============================================================================


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


# =============================================================================
# Supervisor Orchestrator
# =============================================================================


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

        if phase == DeepResearchPhase.PLANNING:
            return {
                **base_inputs,
                "system_prompt": state.system_prompt,
                "max_sub_queries": state.max_sub_queries,
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
                "high_quality_sources": len(
                    [s for s in state.sources if s.quality == SourceQuality.HIGH]
                ),
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
        if phase == DeepResearchPhase.PLANNING:
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
            high_confidence = len(
                [f for f in state.findings if f.confidence == ConfidenceLevel.HIGH]
            )
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
                "next_phase": (
                    DeepResearchPhase.REFINEMENT.value
                    if should_iterate
                    else "COMPLETED"
                ),
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

        state.metadata["agent_decisions"].extend(
            [d.to_dict() for d in self._decisions]
        )
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


# =============================================================================
# Deep Research Workflow
# =============================================================================


class DeepResearchWorkflow(ResearchWorkflowBase):
    """Multi-phase deep research workflow with background execution.

    Supports:
    - Async execution with immediate research_id return
    - Status polling while research runs in background
    - Cancellation and timeout handling
    - Multi-agent supervisor hooks
    - Session persistence for resume capability

    Workflow Phases:
    1. PLANNING - Decompose query into sub-queries
    2. GATHERING - Execute sub-queries in parallel
    3. ANALYSIS - Extract findings and assess quality
    4. SYNTHESIS - Generate comprehensive report
    5. REFINEMENT - Identify gaps and iterate if needed
    """

    # Class-level task registry for background task tracking
    _tasks: WeakValueDictionary[str, BackgroundTask] = WeakValueDictionary()

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
        hooks: Optional[SupervisorHooks] = None,
    ) -> None:
        """Initialize deep research workflow.

        Args:
            config: Research configuration
            memory: Optional memory instance for persistence
            hooks: Optional supervisor hooks for orchestration
        """
        super().__init__(config, memory)
        self.hooks = hooks or SupervisorHooks()
        self.orchestrator = SupervisorOrchestrator()
        self._search_providers: dict[str, SearchProvider] = {}

    def _audit_enabled(self) -> bool:
        """Return True if audit artifacts are enabled."""
        return bool(getattr(self.config, "deep_research_audit_artifacts", True))

    def _audit_path(self, research_id: str) -> Path:
        """Resolve audit artifact path for a research session."""
        # Use memory's base_path which is set from ServerConfig.get_research_dir()
        return self.memory.base_path / "deep_research" / f"{research_id}.audit.jsonl"

    def _write_audit_event(
        self,
        state: Optional[DeepResearchState],
        event_type: str,
        data: Optional[dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """Write a JSONL audit event for deep research observability."""
        if not self._audit_enabled():
            return

        research_id = state.id if state else None
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_id": uuid4().hex,
            "event_type": event_type,
            "level": level,
            "research_id": research_id,
            "phase": state.phase.value if state else None,
            "iteration": state.iteration if state else None,
            "data": data or {},
        }

        try:
            if research_id is None:
                return
            path = self._audit_path(research_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
        except Exception as exc:
            logger.error("Failed to write audit event: %s", exc)
            # Fallback to stderr for crash visibility
            print(
                f"AUDIT_FALLBACK: {event_type} for {research_id} - {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _record_workflow_error(
        self,
        error: Exception,
        state: DeepResearchState,
        context: str,
    ) -> None:
        """Record error to the persistent error store.

        Args:
            error: The exception that occurred
            state: Current research state
            context: Context string (e.g., "background_task", "orchestrator")
        """
        try:
            error_store = FileErrorStore(Path.home() / ".foundry-mcp" / "errors")
            record = ErrorRecord(
                id=f"err_{uuid4().hex[:12]}",
                fingerprint=f"deep-research:{context}:{type(error).__name__}",
                error_code="WORKFLOW_ERROR",
                error_type="internal",
                tool_name=f"deep-research:{context}",
                correlation_id=state.id,
                message=str(error),
                exception_type=type(error).__name__,
                stack_trace=traceback.format_exc(),
                input_summary={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
            )
            error_store.append(record)
        except Exception as store_err:
            logger.error("Failed to record error to store: %s", store_err)

    def _safe_orchestrator_transition(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> None:
        """Safely execute orchestrator phase transition with error logging.

        This wraps orchestrator calls with exception handling to ensure any
        failures are properly logged and recorded before re-raising.

        Args:
            state: Current research state
            phase: The phase that just completed

        Raises:
            Exception: Re-raises any exception after logging
        """
        try:
            self.orchestrator.evaluate_phase_completion(state, phase)
            prompt = self.orchestrator.get_reflection_prompt(state, phase)
            self.hooks.think_pause(state, prompt)
            self.orchestrator.record_to_state(state)
            state.advance_phase()
        except Exception as exc:
            logger.exception(
                "Orchestrator transition failed for phase %s, research %s: %s",
                phase.value,
                state.id,
                exc,
            )
            self._write_audit_event(
                state,
                "orchestrator_error",
                data={
                    "phase": phase.value,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                level="error",
            )
            self._record_workflow_error(exc, state, f"orchestrator_{phase.value}")
            raise  # Re-raise to be caught by workflow exception handler

    # =========================================================================
    # Public API
    # =========================================================================

    def execute(
        self,
        query: Optional[str] = None,
        research_id: Optional[str] = None,
        action: str = "start",
        provider_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 3,
        max_sub_queries: int = 5,
        max_sources_per_query: int = 5,
        follow_links: bool = True,
        timeout_per_operation: float = 120.0,
        max_concurrent: int = 3,
        background: bool = False,
        task_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute deep research workflow.

        Actions:
        - start: Begin new research session
        - continue: Resume existing session
        - status: Get current status
        - report: Get final report
        - cancel: Cancel running task

        Args:
            query: Research query (required for 'start')
            research_id: Session ID (required for continue/status/report/cancel)
            action: One of 'start', 'continue', 'status', 'report', 'cancel'
            provider_id: Provider for LLM operations
            system_prompt: Optional custom system prompt
            max_iterations: Maximum refinement iterations (default: 3)
            max_sub_queries: Maximum sub-queries to generate (default: 5)
            max_sources_per_query: Maximum sources per query (default: 5)
            follow_links: Whether to extract content from URLs (default: True)
            timeout_per_operation: Timeout per operation in seconds (default: 30)
            max_concurrent: Maximum concurrent operations (default: 3)
            background: Run in background, return immediately (default: False)
            task_timeout: Overall timeout for background task (optional)

        Returns:
            WorkflowResult with research state or error
        """
        try:
            if action == "start":
                return self._start_research(
                    query=query,
                    provider_id=provider_id,
                    system_prompt=system_prompt,
                    max_iterations=max_iterations,
                    max_sub_queries=max_sub_queries,
                    max_sources_per_query=max_sources_per_query,
                    follow_links=follow_links,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                    background=background,
                    task_timeout=task_timeout,
                )
            elif action == "continue":
                return self._continue_research(
                    research_id=research_id,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                    background=background,
                    task_timeout=task_timeout,
                )
            elif action == "status":
                return self._get_status(research_id=research_id)
            elif action == "report":
                return self._get_report(research_id=research_id)
            elif action == "cancel":
                return self._cancel_research(research_id=research_id)
            else:
                return WorkflowResult(
                    success=False,
                    content="",
                    error=f"Unknown action '{action}'. Use: start, continue, status, report, cancel",
                )
        except Exception as exc:
            # Catch all exceptions to ensure graceful failure
            logger.exception("Deep research execute failed for action '%s': %s", action, exc)
            return WorkflowResult(
                success=False,
                content="",
                error=f"Deep research failed: {exc}",
                metadata={
                    "action": action,
                    "research_id": research_id,
                    "error_type": exc.__class__.__name__,
                },
            )

    # =========================================================================
    # Background Task Management
    # =========================================================================

    def _start_background_task(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
        task_timeout: Optional[float],
    ) -> WorkflowResult:
        """Start research as a background task using a daemon thread.

        Returns immediately with research_id. The actual workflow
        runs in a daemon thread using asyncio.run().

        This approach works correctly from sync MCP tool handlers where
        there is no running event loop.
        """
        # Create BackgroundTask tracking structure first
        bg_task = BackgroundTask(
            research_id=state.id,
            timeout=task_timeout,
        )
        self._tasks[state.id] = bg_task

        # Register session for crash handler visibility
        _active_research_sessions[state.id] = state

        # Reference to self for use in thread
        workflow = self

        def run_in_thread() -> None:
            """Thread target that runs the async workflow."""
            try:
                async def run_workflow() -> WorkflowResult:
                    """Execute the full workflow asynchronously."""
                    try:
                        coro = workflow._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        )
                        if task_timeout:
                            return await asyncio.wait_for(coro, timeout=task_timeout)
                        return await coro
                    except asyncio.CancelledError:
                        state.metadata["cancelled"] = True
                        workflow.memory.save_deep_research(state)
                        workflow._write_audit_event(
                            state,
                            "workflow_cancelled",
                            data={"cancelled": True},
                            level="warning",
                        )
                        return WorkflowResult(
                            success=False,
                            content="",
                            error="Research was cancelled",
                            metadata={"research_id": state.id, "cancelled": True},
                        )
                    except asyncio.TimeoutError:
                        state.metadata["timeout"] = True
                        state.metadata["abort_phase"] = state.phase.value
                        state.metadata["abort_iteration"] = state.iteration
                        workflow.memory.save_deep_research(state)
                        workflow._write_audit_event(
                            state,
                            "workflow_timeout",
                            data={
                                "timeout_seconds": task_timeout,
                                "abort_phase": state.phase.value,
                                "abort_iteration": state.iteration,
                            },
                            level="warning",
                        )
                        return WorkflowResult(
                            success=False,
                            content="",
                            error=f"Research timed out after {task_timeout}s",
                            metadata={"research_id": state.id, "timeout": True},
                        )
                    except Exception as exc:
                        logger.exception("Background workflow failed: %s", exc)
                        workflow._write_audit_event(
                            state,
                            "workflow_error",
                            data={"error": str(exc)},
                            level="error",
                        )
                        return WorkflowResult(
                            success=False,
                            content="",
                            error=str(exc),
                            metadata={"research_id": state.id},
                        )

                # Run the async workflow in a new event loop
                result = asyncio.run(run_workflow())

                # Handle completion
                if result.metadata and result.metadata.get("timeout"):
                    bg_task.status = TaskStatus.TIMEOUT
                    bg_task.result = result
                    bg_task.completed_at = time.time()
                    bg_task.error = result.error
                else:
                    bg_task.mark_completed(result)

            except Exception as exc:
                # Log the exception with full traceback
                logger.exception(
                    "Background task failed for research %s: %s",
                    state.id, exc
                )
                bg_task.status = TaskStatus.FAILED
                bg_task.error = str(exc)
                bg_task.completed_at = time.time()
                # Record to error store and audit (best effort)
                try:
                    workflow._record_workflow_error(exc, state, "background_task")
                    workflow._write_audit_event(
                        state,
                        "background_task_failed",
                        data={
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        level="error",
                    )
                except Exception:
                    pass  # Already logged above
            finally:
                # Unregister from active sessions
                _active_research_sessions.pop(state.id, None)

        # Create and start the daemon thread
        thread = threading.Thread(
            target=run_in_thread,
            name=f"deep-research-{state.id[:8]}",
            daemon=True,  # Don't prevent process exit
        )
        bg_task.thread = thread

        self._write_audit_event(
            state,
            "background_task_started",
            data={
                "task_timeout": task_timeout,
                "timeout_per_operation": timeout_per_operation,
                "max_concurrent": max_concurrent,
                "thread_name": thread.name,
            },
        )

        thread.start()

        return WorkflowResult(
            success=True,
            content=f"Research started in background: {state.id}",
            metadata={
                "research_id": state.id,
                "background": True,
                "phase": state.phase.value,
            },
        )

    def get_background_task(self, research_id: str) -> Optional[BackgroundTask]:
        """Get a background task by research ID."""
        return self._tasks.get(research_id)

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _start_research(
        self,
        query: Optional[str],
        provider_id: Optional[str],
        system_prompt: Optional[str],
        max_iterations: int,
        max_sub_queries: int,
        max_sources_per_query: int,
        follow_links: bool,
        timeout_per_operation: float,
        max_concurrent: int,
        background: bool,
        task_timeout: Optional[float],
    ) -> WorkflowResult:
        """Start a new deep research session."""
        if not query:
            return WorkflowResult(
                success=False,
                content="",
                error="Query is required to start research",
            )

        # Resolve per-phase providers and models from config
        # Supports ProviderSpec format: "[cli]gemini:pro" -> (provider_id, model)
        planning_pid, planning_model = self.config.resolve_phase_provider("planning")
        analysis_pid, analysis_model = self.config.resolve_phase_provider("analysis")
        synthesis_pid, synthesis_model = self.config.resolve_phase_provider("synthesis")
        refinement_pid, refinement_model = self.config.resolve_phase_provider("refinement")

        # Create initial state with per-phase provider configuration
        state = DeepResearchState(
            original_query=query,
            max_iterations=max_iterations,
            max_sub_queries=max_sub_queries,
            max_sources_per_query=max_sources_per_query,
            follow_links=follow_links,
            research_mode=ResearchMode(self.config.deep_research_mode),
            system_prompt=system_prompt,
            # Per-phase providers: explicit provider_id overrides config
            planning_provider=provider_id or planning_pid,
            analysis_provider=provider_id or analysis_pid,
            synthesis_provider=provider_id or synthesis_pid,
            refinement_provider=provider_id or refinement_pid,
            # Per-phase models from ProviderSpec (only used if provider_id not overridden)
            planning_model=None if provider_id else planning_model,
            analysis_model=None if provider_id else analysis_model,
            synthesis_model=None if provider_id else synthesis_model,
            refinement_model=None if provider_id else refinement_model,
        )

        # Save initial state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "workflow_start",
            data={
                "query": state.original_query,
                "config": {
                    "max_iterations": max_iterations,
                    "max_sub_queries": max_sub_queries,
                    "max_sources_per_query": max_sources_per_query,
                    "follow_links": follow_links,
                    "timeout_per_operation": timeout_per_operation,
                    "max_concurrent": max_concurrent,
                },
                "provider_id": provider_id,
                "background": background,
                "task_timeout": task_timeout,
            },
        )

        if background:
            return self._start_background_task(
                state=state,
                provider_id=provider_id,
                timeout_per_operation=timeout_per_operation,
                max_concurrent=max_concurrent,
                task_timeout=task_timeout,
            )

        # Synchronous execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, run directly
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._execute_workflow_async(
                    state=state,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
            )

    def _continue_research(
        self,
        research_id: Optional[str],
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
        background: bool = False,
        task_timeout: Optional[float] = None,
    ) -> WorkflowResult:
        """Continue an existing research session.

        Args:
            research_id: ID of the research session to continue
            provider_id: Optional provider ID for LLM calls
            timeout_per_operation: Timeout per operation in seconds
            max_concurrent: Maximum concurrent operations
            background: If True, run in background thread (default: False)
            task_timeout: Overall timeout for background task (optional)

        Returns:
            WorkflowResult with research state or error
        """
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required to continue research",
            )

        # Load existing state
        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        if state.completed_at is not None:
            return WorkflowResult(
                success=True,
                content=state.report or "Research already completed",
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "is_complete": True,
                },
            )

        # Run in background if requested
        if background:
            return self._start_background_task(
                state=state,
                provider_id=provider_id,
                timeout_per_operation=timeout_per_operation,
                max_concurrent=max_concurrent,
                task_timeout=task_timeout,
            )

        # Continue from current phase synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        ),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                )
        except RuntimeError:
            return asyncio.run(
                self._execute_workflow_async(
                    state=state,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
            )

    def _get_status(self, research_id: Optional[str]) -> WorkflowResult:
        """Get the current status of a research session."""
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required",
            )

        # Check background task first
        bg_task = self.get_background_task(research_id)
        if bg_task:
            # Also load persisted state to get progress metrics
            state = self.memory.load_deep_research(research_id)
            metadata: dict[str, Any] = {
                "research_id": research_id,
                "task_status": bg_task.status.value,
                "elapsed_ms": bg_task.elapsed_ms,
                "is_complete": bg_task.is_done,
            }
            # Include progress from persisted state if available
            if state:
                metadata.update({
                    "original_query": state.original_query,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "max_iterations": state.max_iterations,
                    "sub_queries_total": len(state.sub_queries),
                    "sub_queries_completed": len(state.completed_sub_queries()),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "total_tokens_used": state.total_tokens_used,
                    "is_failed": bool(state.metadata.get("failed")),
                    "failure_error": state.metadata.get("failure_error"),
                })
            return WorkflowResult(
                success=True,
                content=f"Task status: {bg_task.status.value}",
                metadata=metadata,
            )

        # Fall back to persisted state (task completed or not running)
        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        # Determine status string
        is_failed = bool(state.metadata.get("failed"))
        if is_failed:
            status_str = "Failed"
        elif state.completed_at:
            status_str = "Completed"
        else:
            status_str = "In Progress"

        status_lines = [
            f"Research ID: {state.id}",
            f"Query: {state.original_query}",
            f"Phase: {state.phase.value}",
            f"Iteration: {state.iteration}/{state.max_iterations}",
            f"Sub-queries: {len(state.completed_sub_queries())}/{len(state.sub_queries)} completed",
            f"Sources: {len(state.sources)} examined",
            f"Findings: {len(state.findings)}",
            f"Gaps: {len(state.unresolved_gaps())} unresolved",
            f"Status: {status_str}",
        ]
        if state.metadata.get("timeout"):
            status_lines.append("Timeout: True")
        if state.metadata.get("cancelled"):
            status_lines.append("Cancelled: True")
        if is_failed:
            failure_error = state.metadata.get("failure_error", "Unknown error")
            status_lines.append(f"Error: {failure_error}")

        return WorkflowResult(
            success=True,
            content="\n".join(status_lines),
            metadata={
                "research_id": state.id,
                "original_query": state.original_query,
                "phase": state.phase.value,
                "iteration": state.iteration,
                "max_iterations": state.max_iterations,
                "sub_queries_total": len(state.sub_queries),
                "sub_queries_completed": len(state.completed_sub_queries()),
                "source_count": len(state.sources),
                "finding_count": len(state.findings),
                "gap_count": len(state.unresolved_gaps()),
                "is_complete": state.completed_at is not None,
                "is_failed": is_failed,
                "failure_error": state.metadata.get("failure_error"),
                "total_tokens_used": state.total_tokens_used,
                "total_duration_ms": state.total_duration_ms,
                "timed_out": bool(state.metadata.get("timeout")),
                "cancelled": bool(state.metadata.get("cancelled")),
            },
        )

    def _get_report(self, research_id: Optional[str]) -> WorkflowResult:
        """Get the final report from a research session."""
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required",
            )

        state = self.memory.load_deep_research(research_id)
        if state is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found",
            )

        if not state.report:
            return WorkflowResult(
                success=False,
                content="",
                error="Research report not yet generated",
            )

        return WorkflowResult(
            success=True,
            content=state.report,
            metadata={
                "research_id": state.id,
                "original_query": state.original_query,
                "source_count": len(state.sources),
                "finding_count": len(state.findings),
                "iteration": state.iteration,
                "is_complete": state.completed_at is not None,
            },
        )

    def _cancel_research(self, research_id: Optional[str]) -> WorkflowResult:
        """Cancel a running research task."""
        if not research_id:
            return WorkflowResult(
                success=False,
                content="",
                error="research_id is required",
            )

        bg_task = self.get_background_task(research_id)
        if bg_task is None:
            return WorkflowResult(
                success=False,
                content="",
                error=f"No running task found for '{research_id}'",
            )

        if bg_task.cancel():
            state = self.memory.load_deep_research(research_id)
            if state:
                self._write_audit_event(
                    state,
                    "workflow_cancelled",
                    data={"cancelled": True},
                    level="warning",
                )
            return WorkflowResult(
                success=True,
                content=f"Research '{research_id}' cancelled",
                metadata={"research_id": research_id, "cancelled": True},
            )
        else:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Task '{research_id}' already completed",
            )

    # =========================================================================
    # Async Workflow Execution
    # =========================================================================

    async def _execute_workflow_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
    ) -> WorkflowResult:
        """Execute the full workflow asynchronously.

        This is the main async entry point that orchestrates all phases.
        """
        start_time = time.perf_counter()

        try:
            # Phase execution based on current state
            if state.phase == DeepResearchPhase.PLANNING:
                phase_started = time.perf_counter()
                self.hooks.emit_phase_start(state)
                self._write_audit_event(
                    state,
                    "phase_start",
                    data={"phase": state.phase.value},
                )
                result = await self._execute_planning_async(
                    state=state,
                    provider_id=state.planning_provider,
                    timeout=self.config.get_phase_timeout("planning"),
                )
                if not result.success:
                    self._write_audit_event(
                        state,
                        "phase_error",
                        data={"phase": state.phase.value, "error": result.error},
                        level="error",
                    )
                    state.mark_failed(result.error or f"Phase {state.phase.value} failed")
                    self.memory.save_deep_research(state)
                    return result
                self.hooks.emit_phase_complete(state)
                self._write_audit_event(
                    state,
                    "phase_complete",
                    data={
                        "phase": state.phase.value,
                        "duration_ms": (time.perf_counter() - phase_started) * 1000,
                    },
                )
                # Think pause: supervisor evaluates planning quality
                self._safe_orchestrator_transition(state, DeepResearchPhase.PLANNING)

            if state.phase == DeepResearchPhase.GATHERING:
                phase_started = time.perf_counter()
                self.hooks.emit_phase_start(state)
                self._write_audit_event(
                    state,
                    "phase_start",
                    data={"phase": state.phase.value},
                )
                result = await self._execute_gathering_async(
                    state=state,
                    provider_id=provider_id,
                    timeout=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
                if not result.success:
                    self._write_audit_event(
                        state,
                        "phase_error",
                        data={"phase": state.phase.value, "error": result.error},
                        level="error",
                    )
                    state.mark_failed(result.error or f"Phase {state.phase.value} failed")
                    self.memory.save_deep_research(state)
                    return result
                self.hooks.emit_phase_complete(state)
                self._write_audit_event(
                    state,
                    "phase_complete",
                    data={
                        "phase": state.phase.value,
                        "duration_ms": (time.perf_counter() - phase_started) * 1000,
                    },
                )
                # Think pause: supervisor evaluates gathering quality
                self._safe_orchestrator_transition(state, DeepResearchPhase.GATHERING)

            if state.phase == DeepResearchPhase.ANALYSIS:
                phase_started = time.perf_counter()
                self.hooks.emit_phase_start(state)
                self._write_audit_event(
                    state,
                    "phase_start",
                    data={"phase": state.phase.value},
                )
                result = await self._execute_analysis_async(
                    state=state,
                    provider_id=state.analysis_provider,
                    timeout=self.config.get_phase_timeout("analysis"),
                )
                if not result.success:
                    self._write_audit_event(
                        state,
                        "phase_error",
                        data={"phase": state.phase.value, "error": result.error},
                        level="error",
                    )
                    state.mark_failed(result.error or f"Phase {state.phase.value} failed")
                    self.memory.save_deep_research(state)
                    return result
                self.hooks.emit_phase_complete(state)
                self._write_audit_event(
                    state,
                    "phase_complete",
                    data={
                        "phase": state.phase.value,
                        "duration_ms": (time.perf_counter() - phase_started) * 1000,
                    },
                )
                # Think pause: supervisor evaluates analysis quality
                self._safe_orchestrator_transition(state, DeepResearchPhase.ANALYSIS)

            if state.phase == DeepResearchPhase.SYNTHESIS:
                phase_started = time.perf_counter()
                self.hooks.emit_phase_start(state)
                self._write_audit_event(
                    state,
                    "phase_start",
                    data={"phase": state.phase.value},
                )
                result = await self._execute_synthesis_async(
                    state=state,
                    provider_id=state.synthesis_provider,
                    timeout=self.config.get_phase_timeout("synthesis"),
                )
                if not result.success:
                    self._write_audit_event(
                        state,
                        "phase_error",
                        data={"phase": state.phase.value, "error": result.error},
                        level="error",
                    )
                    state.mark_failed(result.error or f"Phase {state.phase.value} failed")
                    self.memory.save_deep_research(state)
                    return result
                self.hooks.emit_phase_complete(state)
                self._write_audit_event(
                    state,
                    "phase_complete",
                    data={
                        "phase": state.phase.value,
                        "duration_ms": (time.perf_counter() - phase_started) * 1000,
                    },
                )
                # Think pause: supervisor evaluates synthesis and decides iteration
                try:
                    self.orchestrator.evaluate_phase_completion(state, DeepResearchPhase.SYNTHESIS)
                    self.orchestrator.decide_iteration(state)
                    prompt = self.orchestrator.get_reflection_prompt(state, DeepResearchPhase.SYNTHESIS)
                    self.hooks.think_pause(state, prompt)
                    self.orchestrator.record_to_state(state)
                except Exception as exc:
                    logger.exception(
                        "Orchestrator transition failed for synthesis, research %s: %s",
                        state.id,
                        exc,
                    )
                    self._write_audit_event(
                        state,
                        "orchestrator_error",
                        data={
                            "phase": "synthesis",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        level="error",
                    )
                    self._record_workflow_error(exc, state, "orchestrator_synthesis")
                    raise

                # Check if refinement needed
                if state.should_continue_refinement():
                    state.phase = DeepResearchPhase.REFINEMENT
                else:
                    state.mark_completed(report=result.content)

            # Handle refinement phase
            if state.phase == DeepResearchPhase.REFINEMENT:
                phase_started = time.perf_counter()
                self.hooks.emit_phase_start(state)
                self._write_audit_event(
                    state,
                    "phase_start",
                    data={"phase": state.phase.value},
                )
                # Generate follow-up queries from gaps
                await self._execute_refinement_async(
                    state=state,
                    provider_id=state.refinement_provider,
                    timeout=self.config.get_phase_timeout("refinement"),
                )
                self.hooks.emit_phase_complete(state)
                self._write_audit_event(
                    state,
                    "phase_complete",
                    data={
                        "phase": state.phase.value,
                        "duration_ms": (time.perf_counter() - phase_started) * 1000,
                    },
                )

                if state.should_continue_refinement():
                    state.start_new_iteration()
                    # Recursively continue workflow
                    return await self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                else:
                    state.mark_completed(report=state.report)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            state.total_duration_ms += duration_ms

            # Save final state
            self.memory.save_deep_research(state)
            self._write_audit_event(
                state,
                "workflow_complete",
                data={
                    "success": True,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "sub_query_count": len(state.sub_queries),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "report_length": len(state.report or ""),
                    # Existing totals
                    "total_tokens_used": state.total_tokens_used,
                    "total_duration_ms": state.total_duration_ms,
                    # Token breakdown totals
                    "total_input_tokens": sum(
                        m.input_tokens for m in state.phase_metrics
                    ),
                    "total_output_tokens": sum(
                        m.output_tokens for m in state.phase_metrics
                    ),
                    "total_cached_tokens": sum(
                        m.cached_tokens for m in state.phase_metrics
                    ),
                    # Per-phase metrics
                    "phase_metrics": [
                        {
                            "phase": m.phase,
                            "duration_ms": m.duration_ms,
                            "input_tokens": m.input_tokens,
                            "output_tokens": m.output_tokens,
                            "cached_tokens": m.cached_tokens,
                            "provider_id": m.provider_id,
                            "model_used": m.model_used,
                        }
                        for m in state.phase_metrics
                    ],
                    # Search provider stats
                    "search_provider_stats": state.search_provider_stats,
                    "total_search_queries": sum(state.search_provider_stats.values()),
                    # Source hostnames
                    "source_hostnames": sorted(
                        set(
                            h
                            for s in state.sources
                            if s.url and (h := _extract_hostname(s.url))
                        )
                    ),
                    # Research mode
                    "research_mode": state.research_mode.value,
                },
            )

            return WorkflowResult(
                success=True,
                content=state.report or "Research completed",
                provider_id=provider_id,
                tokens_used=state.total_tokens_used,
                duration_ms=duration_ms,
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "sub_query_count": len(state.sub_queries),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "is_complete": state.completed_at is not None,
                },
            )

        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.exception(
                "Workflow execution failed at phase %s, iteration %d: %s",
                state.phase.value,
                state.iteration,
                exc,
            )
            self.memory.save_deep_research(state)
            self._write_audit_event(
                state,
                "workflow_error",
                data={
                    "error": str(exc),
                    "traceback": tb_str,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
                level="error",
            )
            self._record_workflow_error(exc, state, "workflow_execution")
            return WorkflowResult(
                success=False,
                content="",
                error=str(exc),
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
            )

    # =========================================================================
    # Phase Implementations (Stubs for now - implemented in later tasks)
    # =========================================================================

    async def _execute_planning_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute planning phase: decompose query into sub-queries.

        This phase:
        1. Analyzes the original research query
        2. Generates a research brief explaining the approach
        3. Decomposes the query into 2-5 focused sub-queries
        4. Assigns priorities to each sub-query

        Args:
            state: Current research state
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with planning outcome
        """
        logger.info("Starting planning phase for query: %s", state.original_query[:100])

        # Build the planning prompt
        system_prompt = self._build_planning_system_prompt(state)
        user_prompt = self._build_planning_user_prompt(state)

        # Execute LLM call with context window error handling and timeout protection
        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id or state.planning_provider,
                model=state.planning_model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.7,  # Some creativity for diverse sub-queries
                phase="planning",
                fallback_providers=self.config.get_phase_fallback_providers("planning"),
                max_retries=self.config.deep_research_max_retries,
                retry_delay=self.config.deep_research_retry_delay,
            )
        except ContextWindowError as e:
            logger.error(
                "Planning phase context window exceeded: prompt_tokens=%s, "
                "max_tokens=%s, truncation_needed=%s, provider=%s",
                e.prompt_tokens,
                e.max_tokens,
                e.truncation_needed,
                e.provider,
            )
            return WorkflowResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "research_id": state.id,
                    "phase": "planning",
                    "error_type": "context_window_exceeded",
                    "prompt_tokens": e.prompt_tokens,
                    "max_tokens": e.max_tokens,
                    "truncation_needed": e.truncation_needed,
                },
            )

        if not result.success:
            # Check if this was a timeout
            if result.metadata and result.metadata.get("timeout"):
                logger.error(
                    "Planning phase timed out after exhausting all providers: %s",
                    result.metadata.get("providers_tried", []),
                )
            else:
                logger.error("Planning phase LLM call failed: %s", result.error)
            return result

        # Track token usage
        if result.tokens_used:
            state.total_tokens_used += result.tokens_used

        # Track phase metrics for audit
        state.phase_metrics.append(
            PhaseMetrics(
                phase="planning",
                duration_ms=result.duration_ms or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                provider_id=result.provider_id,
                model_used=result.model_used,
            )
        )

        # Parse the response
        parsed = self._parse_planning_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse planning response, using fallback")
            # Fallback: treat entire query as single sub-query
            state.research_brief = f"Direct research on: {state.original_query}"
            state.add_sub_query(
                query=state.original_query,
                rationale="Original query used directly due to parsing failure",
                priority=1,
            )
        else:
            state.research_brief = parsed["research_brief"]
            for sq in parsed["sub_queries"]:
                state.add_sub_query(
                    query=sq["query"],
                    rationale=sq.get("rationale"),
                    priority=sq.get("priority", 1),
                )

        # Save state after planning
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "planning_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "parse_success": parsed["success"],
                "research_brief": state.research_brief,
                "sub_queries": [
                    {
                        "id": sq.id,
                        "query": sq.query,
                        "rationale": sq.rationale,
                        "priority": sq.priority,
                    }
                    for sq in state.sub_queries
                ],
            },
        )

        logger.info(
            "Planning phase complete: %d sub-queries generated",
            len(state.sub_queries),
        )

        return WorkflowResult(
            success=True,
            content=state.research_brief or "Planning complete",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "sub_query_count": len(state.sub_queries),
                "research_brief": state.research_brief,
            },
        )

    def _build_planning_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for query decomposition.

        Args:
            state: Current research state

        Returns:
            System prompt string
        """
        return """You are a research planning assistant. Your task is to analyze a research query and decompose it into focused sub-queries that can be researched independently.

Your response MUST be valid JSON with this exact structure:
{
    "research_brief": "A 2-3 sentence summary of the research approach and what aspects will be investigated",
    "sub_queries": [
        {
            "query": "A specific, focused search query",
            "rationale": "Why this sub-query is important for the research",
            "priority": 1
        }
    ]
}

Guidelines:
- Generate 2-5 sub-queries (aim for 3-4 typically)
- Each sub-query should focus on a distinct aspect of the research
- Queries should be specific enough to yield relevant search results
- Priority 1 is highest (most important), higher numbers are lower priority
- Avoid overlapping queries - each should cover unique ground
- Consider different angles: definition, examples, comparisons, recent developments, expert opinions

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_planning_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt for query decomposition.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        prompt = f"""Research Query: {state.original_query}

Please decompose this research query into {state.max_sub_queries} or fewer focused sub-queries.

Consider:
1. What are the key aspects that need investigation?
2. What background information would help understand this topic?
3. What specific questions would lead to comprehensive coverage?
4. What different perspectives or sources might be valuable?

Generate the research plan as JSON."""

        # Add custom system prompt context if provided
        if state.system_prompt:
            prompt += f"\n\nAdditional context: {state.system_prompt}"

        return prompt

    def _parse_planning_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured planning data.

        Attempts to extract JSON from the response, with fallback handling
        for various response formats.

        Args:
            content: Raw LLM response content
            state: Current research state (for max_sub_queries limit)

        Returns:
            Dict with 'success', 'research_brief', and 'sub_queries' keys
        """
        result = {
            "success": False,
            "research_brief": None,
            "sub_queries": [],
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = self._extract_json(content)
        if not json_str:
            logger.warning("No JSON found in planning response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from planning response: %s", e)
            return result

        # Extract research brief
        result["research_brief"] = data.get("research_brief", "")

        # Extract and validate sub-queries
        raw_queries = data.get("sub_queries", [])
        if not isinstance(raw_queries, list):
            logger.warning("sub_queries is not a list")
            return result

        for i, sq in enumerate(raw_queries):
            if not isinstance(sq, dict):
                continue
            query = sq.get("query", "").strip()
            if not query:
                continue

            # Limit to max_sub_queries
            if len(result["sub_queries"]) >= state.max_sub_queries:
                break

            result["sub_queries"].append({
                "query": query,
                "rationale": sq.get("rationale", ""),
                "priority": min(max(int(sq.get("priority", i + 1)), 1), 10),
            })

        # Mark success if we got at least one sub-query
        result["success"] = len(result["sub_queries"]) > 0

        return result

    def _extract_json(self, content: str) -> Optional[str]:
        """Extract JSON object from content that may contain other text.

        Handles cases where JSON is wrapped in markdown code blocks
        or mixed with explanatory text.

        Args:
            content: Raw content that may contain JSON

        Returns:
            Extracted JSON string or None if not found
        """
        # First, try to find JSON in code blocks
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, content)
        for match in matches:
            match = match.strip()
            if match.startswith('{'):
                return match

        # Try to find raw JSON object
        # Look for the outermost { ... } pair
        brace_start = content.find('{')
        if brace_start == -1:
            return None

        # Find matching closing brace
        depth = 0
        for i, char in enumerate(content[brace_start:], brace_start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return content[brace_start:i + 1]

        return None

    async def _execute_gathering_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
        max_concurrent: int,
    ) -> WorkflowResult:
        """Execute gathering phase: parallel sub-query execution.

        This phase:
        1. Gets all pending sub-queries from planning phase
        2. Executes them concurrently with rate limiting
        3. Collects and deduplicates sources
        4. Marks sub-queries as completed/failed

        Args:
            state: Current research state with sub-queries
            provider_id: LLM provider (not used in gathering)
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent search requests

        Returns:
            WorkflowResult with gathering outcome
        """
        pending_queries = state.pending_sub_queries()
        if not pending_queries:
            logger.warning("No pending sub-queries for gathering phase")
            return WorkflowResult(
                success=True,
                content="No sub-queries to execute",
                metadata={"research_id": state.id, "source_count": 0},
            )

        logger.info(
            "Starting gathering phase: %d sub-queries, max_concurrent=%d",
            len(pending_queries),
            max_concurrent,
        )

        provider_names = getattr(
            self.config,
            "deep_research_providers",
            ["tavily", "google", "semantic_scholar"],
        )
        available_providers: list[SearchProvider] = []
        unavailable_providers: list[str] = []

        for name in provider_names:
            provider = self._get_search_provider(name)
            if provider is None:
                unavailable_providers.append(name)
                continue
            available_providers.append(provider)

        if not available_providers:
            return WorkflowResult(
                success=False,
                content="",
                error=(
                    "No search providers available. Configure API keys for "
                    "Tavily, Google, or Semantic Scholar."
                ),
            )

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Track collected sources for deduplication
        seen_urls: set[str] = set()
        seen_titles: dict[str, str] = {}  # normalized_title -> first source URL
        total_sources_added = 0
        failed_queries = 0

        async def execute_sub_query(sub_query) -> tuple[int, Optional[str]]:
            """Execute a single sub-query and return (sources_added, error)."""
            async with semaphore:
                sub_query.status = "executing"

                provider_errors: list[str] = []
                added = 0

                for provider in available_providers:
                    provider_name = provider.get_provider_name()
                    try:
                        sources = await provider.search(
                            query=sub_query.query,
                            max_results=state.max_sources_per_query,
                            sub_query_id=sub_query.id,
                            include_raw_content=state.follow_links,
                        )

                        # Add sources with deduplication
                        for source in sources:
                            # URL-based deduplication
                            if source.url and source.url in seen_urls:
                                continue  # Skip duplicate URL

                            # Title-based deduplication (same paper from different domains)
                            normalized_title = _normalize_title(source.title)
                            if normalized_title and len(normalized_title) > 20:
                                if normalized_title in seen_titles:
                                    logger.debug(
                                        "Skipping duplicate by title: %s (already have %s)",
                                        source.url,
                                        seen_titles[normalized_title],
                                    )
                                    continue  # Skip duplicate title
                                seen_titles[normalized_title] = source.url or ""

                            if source.url:
                                seen_urls.add(source.url)
                                # Apply domain-based quality scoring
                                if source.quality == SourceQuality.UNKNOWN:
                                    source.quality = get_domain_quality(
                                        source.url, state.research_mode
                                    )

                            # Add source to state
                            state.sources.append(source)
                            state.total_sources_examined += 1
                            sub_query.source_ids.append(source.id)
                            added += 1

                        self._write_audit_event(
                            state,
                            "gathering_provider_result",
                            data={
                                "provider": provider_name,
                                "sub_query_id": sub_query.id,
                                "sub_query": sub_query.query,
                                "sources_added": len(sources),
                            },
                        )
                        # Track search provider query count
                        state.search_provider_stats[provider_name] = (
                            state.search_provider_stats.get(provider_name, 0) + 1
                        )
                    except SearchProviderError as e:
                        provider_errors.append(f"{provider_name}: {e}")
                        self._write_audit_event(
                            state,
                            "gathering_provider_result",
                            data={
                                "provider": provider_name,
                                "sub_query_id": sub_query.id,
                                "sub_query": sub_query.query,
                                "sources_added": 0,
                                "error": str(e),
                            },
                            level="warning",
                        )
                    except Exception as e:
                        provider_errors.append(f"{provider_name}: {e}")
                        self._write_audit_event(
                            state,
                            "gathering_provider_result",
                            data={
                                "provider": provider_name,
                                "sub_query_id": sub_query.id,
                                "sub_query": sub_query.query,
                                "sources_added": 0,
                                "error": str(e),
                            },
                            level="warning",
                        )

                if added > 0:
                    sub_query.mark_completed(
                        findings=f"Found {added} sources"
                    )
                    logger.debug(
                        "Sub-query '%s' completed: %d sources",
                        sub_query.query[:50],
                        added,
                    )
                    return added, None

                error_summary = "; ".join(provider_errors) or "No sources found"
                sub_query.mark_failed(error_summary)
                logger.warning(
                    "Sub-query '%s' failed: %s",
                    sub_query.query[:50],
                    error_summary,
                )
                return 0, error_summary

        # Execute all sub-queries concurrently
        tasks = [execute_sub_query(sq) for sq in pending_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                failed_queries += 1
                logger.error("Task exception: %s", result)
            else:
                added, error = result
                total_sources_added += added
                if error:
                    failed_queries += 1

        # Update state timestamp
        state.updated_at = __import__("datetime").datetime.utcnow()

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "gathering_result",
            data={
                "source_count": total_sources_added,
                "queries_executed": len(pending_queries),
                "queries_failed": failed_queries,
                "unique_urls": len(seen_urls),
                "providers_used": [p.get_provider_name() for p in available_providers],
                "providers_unavailable": unavailable_providers,
            },
        )

        # Determine success
        success = total_sources_added > 0 or failed_queries < len(pending_queries)

        # Build error message if all queries failed
        error_msg = None
        if not success:
            providers_used = [p.get_provider_name() for p in available_providers]
            if failed_queries == len(pending_queries):
                error_msg = (
                    f"All {failed_queries} sub-queries failed to find sources. "
                    f"Providers used: {providers_used}. "
                    f"Unavailable providers: {unavailable_providers}"
                )

        logger.info(
            "Gathering phase complete: %d sources from %d queries (%d failed)",
            total_sources_added,
            len(pending_queries),
            failed_queries,
        )

        return WorkflowResult(
            success=success,
            content=f"Gathered {total_sources_added} sources from {len(pending_queries)} sub-queries",
            error=error_msg,
            metadata={
                "research_id": state.id,
                "source_count": total_sources_added,
                "queries_executed": len(pending_queries),
                "queries_failed": failed_queries,
                "unique_urls": len(seen_urls),
                "providers_used": [p.get_provider_name() for p in available_providers],
                "providers_unavailable": unavailable_providers,
            },
        )

    def _get_search_provider(self, provider_name: str) -> Optional[SearchProvider]:
        """Get or create a search provider instance.

        Args:
            provider_name: Name of the provider (e.g., "tavily")

        Returns:
            SearchProvider instance or None if unavailable
        """
        if provider_name in self._search_providers:
            return self._search_providers[provider_name]

        try:
            if provider_name == "tavily":
                provider = TavilySearchProvider()
                self._search_providers[provider_name] = provider
                return provider
            if provider_name == "perplexity":
                provider = PerplexitySearchProvider()
                self._search_providers[provider_name] = provider
                return provider
            if provider_name == "google":
                provider = GoogleSearchProvider()
                self._search_providers[provider_name] = provider
                return provider
            if provider_name == "semantic_scholar":
                provider = SemanticScholarProvider()
                self._search_providers[provider_name] = provider
                return provider
            else:
                logger.warning("Unknown search provider: %s", provider_name)
                return None
        except ValueError as e:
            # API key not configured
            logger.error("Failed to initialize %s provider: %s", provider_name, e)
            return None
        except Exception as e:
            logger.error("Error initializing %s provider: %s", provider_name, e)
            return None

    async def _execute_analysis_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute analysis phase: extract findings from sources.

        This phase:
        1. Builds prompt with gathered source summaries
        2. Uses LLM to extract key findings
        3. Assesses confidence levels for each finding
        4. Identifies knowledge gaps requiring follow-up
        5. Updates source quality assessments

        Args:
            state: Current research state with gathered sources
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with analysis outcome
        """
        if not state.sources:
            logger.warning("No sources to analyze")
            return WorkflowResult(
                success=True,
                content="No sources to analyze",
                metadata={"research_id": state.id, "finding_count": 0},
            )

        logger.info(
            "Starting analysis phase: %d sources to analyze",
            len(state.sources),
        )

        # Build the analysis prompt
        system_prompt = self._build_analysis_system_prompt(state)
        user_prompt = self._build_analysis_user_prompt(state)

        # Execute LLM call with context window error handling and timeout protection
        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id or state.analysis_provider,
                model=state.analysis_model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.3,  # Lower temperature for analytical tasks
                phase="analysis",
                fallback_providers=self.config.get_phase_fallback_providers("analysis"),
                max_retries=self.config.deep_research_max_retries,
                retry_delay=self.config.deep_research_retry_delay,
            )
        except ContextWindowError as e:
            logger.error(
                "Analysis phase context window exceeded: prompt_tokens=%s, "
                "max_tokens=%s, truncation_needed=%s, provider=%s, source_count=%d",
                e.prompt_tokens,
                e.max_tokens,
                e.truncation_needed,
                e.provider,
                len(state.sources),
            )
            return WorkflowResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "research_id": state.id,
                    "phase": "analysis",
                    "error_type": "context_window_exceeded",
                    "prompt_tokens": e.prompt_tokens,
                    "max_tokens": e.max_tokens,
                    "truncation_needed": e.truncation_needed,
                    "source_count": len(state.sources),
                    "guidance": "Try reducing max_sources_per_query or processing sources in batches",
                },
            )

        if not result.success:
            # Check if this was a timeout
            if result.metadata and result.metadata.get("timeout"):
                logger.error(
                    "Analysis phase timed out after exhausting all providers: %s",
                    result.metadata.get("providers_tried", []),
                )
            else:
                logger.error("Analysis phase LLM call failed: %s", result.error)
            return result

        # Track token usage
        if result.tokens_used:
            state.total_tokens_used += result.tokens_used

        # Track phase metrics for audit
        state.phase_metrics.append(
            PhaseMetrics(
                phase="analysis",
                duration_ms=result.duration_ms or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                provider_id=result.provider_id,
                model_used=result.model_used,
            )
        )

        # Parse the response
        parsed = self._parse_analysis_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse analysis response")
            self._write_audit_event(
                state,
                "analysis_result",
                data={
                    "provider_id": result.provider_id,
                    "model_used": result.model_used,
                    "tokens_used": result.tokens_used,
                    "duration_ms": result.duration_ms,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": result.content,
                    "parse_success": False,
                    "findings": [],
                    "gaps": [],
                    "quality_updates": [],
                },
                level="warning",
            )
            # Still mark as success but with no findings
            return WorkflowResult(
                success=True,
                content="Analysis completed but no findings extracted",
                metadata={
                    "research_id": state.id,
                    "finding_count": 0,
                    "parse_error": True,
                },
            )

        # Add findings to state
        for finding_data in parsed["findings"]:
            state.add_finding(
                content=finding_data["content"],
                confidence=finding_data["confidence"],
                source_ids=finding_data.get("source_ids", []),
                category=finding_data.get("category"),
            )

        # Add gaps to state
        for gap_data in parsed["gaps"]:
            state.add_gap(
                description=gap_data["description"],
                suggested_queries=gap_data.get("suggested_queries", []),
                priority=gap_data.get("priority", 1),
            )

        # Update source quality assessments
        for quality_update in parsed.get("quality_updates", []):
            source = state.get_source(quality_update["source_id"])
            if source:
                try:
                    source.quality = SourceQuality(quality_update["quality"])
                except ValueError:
                    pass  # Invalid quality value, skip

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "analysis_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "parse_success": True,
                "findings": parsed["findings"],
                "gaps": parsed["gaps"],
                "quality_updates": parsed.get("quality_updates", []),
            },
        )

        logger.info(
            "Analysis phase complete: %d findings, %d gaps identified",
            len(parsed["findings"]),
            len(parsed["gaps"]),
        )

        return WorkflowResult(
            success=True,
            content=f"Extracted {len(parsed['findings'])} findings and identified {len(parsed['gaps'])} gaps",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(parsed["findings"]),
                "gap_count": len(parsed["gaps"]),
                "source_count": len(state.sources),
            },
        )

    def _build_analysis_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for source analysis.

        Args:
            state: Current research state

        Returns:
            System prompt string
        """
        return """You are a research analyst. Your task is to analyze research sources and extract key findings, assess their quality, and identify knowledge gaps.

Your response MUST be valid JSON with this exact structure:
{
    "findings": [
        {
            "content": "A clear, specific finding or insight extracted from the sources",
            "confidence": "low|medium|high",
            "source_ids": ["src-xxx", "src-yyy"],
            "category": "optional category/theme"
        }
    ],
    "gaps": [
        {
            "description": "Description of missing information or unanswered question",
            "suggested_queries": ["follow-up query 1", "follow-up query 2"],
            "priority": 1
        }
    ],
    "quality_updates": [
        {
            "source_id": "src-xxx",
            "quality": "low|medium|high"
        }
    ]
}

Guidelines for findings:
- Extract 2-5 key findings from the sources
- Each finding should be a specific, actionable insight
- Confidence levels: "low" (single weak source), "medium" (multiple sources or one authoritative), "high" (multiple authoritative sources agree)
- Include source_ids that support each finding
- Categorize findings by theme when applicable

Guidelines for gaps:
- Identify 1-3 knowledge gaps or unanswered questions
- Provide specific follow-up queries that could fill each gap
- Priority 1 is most important, higher numbers are lower priority

Guidelines for quality_updates:
- Assess source quality based on authority, relevance, and recency
- "low" = questionable reliability, "medium" = generally reliable, "high" = authoritative

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_analysis_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt with source summaries for analysis.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"Original Research Query: {state.original_query}",
            "",
            "Research Brief:",
            state.research_brief or "Direct research on the query",
            "",
            "Sources to Analyze:",
            "",
        ]

        # Add source summaries
        for i, source in enumerate(state.sources[:20], 1):  # Limit to 20 sources
            prompt_parts.append(f"Source {i} (ID: {source.id}):")
            prompt_parts.append(f"  Title: {source.title}")
            if source.url:
                prompt_parts.append(f"  URL: {source.url}")
            if source.snippet:
                # Truncate long snippets
                snippet = source.snippet[:500] + "..." if len(source.snippet) > 500 else source.snippet
                prompt_parts.append(f"  Snippet: {snippet}")
            if source.content:
                # Truncate long content
                content = source.content[:1000] + "..." if len(source.content) > 1000 else source.content
                prompt_parts.append(f"  Content: {content}")
            prompt_parts.append("")

        prompt_parts.extend([
            "Please analyze these sources and:",
            "1. Extract 2-5 key findings relevant to the research query",
            "2. Assess confidence levels based on source agreement and authority",
            "3. Identify any knowledge gaps or unanswered questions",
            "4. Assess the quality of each source",
            "",
            "Return your analysis as JSON.",
        ])

        return "\n".join(prompt_parts)

    def _parse_analysis_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured analysis data.

        Args:
            content: Raw LLM response content
            state: Current research state

        Returns:
            Dict with 'success', 'findings', 'gaps', and 'quality_updates' keys
        """
        result = {
            "success": False,
            "findings": [],
            "gaps": [],
            "quality_updates": [],
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = self._extract_json(content)
        if not json_str:
            logger.warning("No JSON found in analysis response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from analysis response: %s", e)
            return result

        # Parse findings
        raw_findings = data.get("findings", [])
        if isinstance(raw_findings, list):
            for f in raw_findings:
                if not isinstance(f, dict):
                    continue
                content_text = f.get("content", "").strip()
                if not content_text:
                    continue

                # Map confidence string to enum
                confidence_str = f.get("confidence", "medium").lower()
                confidence_map = {
                    "low": ConfidenceLevel.LOW,
                    "medium": ConfidenceLevel.MEDIUM,
                    "high": ConfidenceLevel.HIGH,
                    "confirmed": ConfidenceLevel.CONFIRMED,
                    "speculation": ConfidenceLevel.SPECULATION,
                }
                confidence = confidence_map.get(confidence_str, ConfidenceLevel.MEDIUM)

                result["findings"].append({
                    "content": content_text,
                    "confidence": confidence,
                    "source_ids": f.get("source_ids", []),
                    "category": f.get("category"),
                })

        # Parse gaps
        raw_gaps = data.get("gaps", [])
        if isinstance(raw_gaps, list):
            for g in raw_gaps:
                if not isinstance(g, dict):
                    continue
                description = g.get("description", "").strip()
                if not description:
                    continue

                result["gaps"].append({
                    "description": description,
                    "suggested_queries": g.get("suggested_queries", []),
                    "priority": min(max(int(g.get("priority", 1)), 1), 10),
                })

        # Parse quality updates
        raw_quality = data.get("quality_updates", [])
        if isinstance(raw_quality, list):
            for q in raw_quality:
                if not isinstance(q, dict):
                    continue
                source_id = q.get("source_id", "").strip()
                quality = q.get("quality", "").lower()
                if source_id and quality in ("low", "medium", "high", "unknown"):
                    result["quality_updates"].append({
                        "source_id": source_id,
                        "quality": quality,
                    })

        # Mark success if we got at least one finding
        result["success"] = len(result["findings"]) > 0

        return result

    async def _execute_synthesis_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute synthesis phase: generate comprehensive report from findings.

        This phase:
        1. Builds a synthesis prompt with all findings grouped by theme
        2. Includes source references for citation
        3. Generates a structured markdown report with:
           - Executive summary
           - Key findings organized by theme
           - Source citations
           - Knowledge gaps and limitations
           - Conclusions with actionable insights
        4. Stores the report in state.report

        Args:
            state: Current research state with findings from analysis
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with synthesis outcome
        """
        if not state.findings:
            logger.warning("No findings to synthesize")
            # Generate a minimal report even without findings
            state.report = self._generate_empty_report(state)
            self._write_audit_event(
                state,
                "synthesis_result",
                data={
                    "provider_id": None,
                    "model_used": None,
                    "tokens_used": None,
                    "duration_ms": None,
                    "system_prompt": None,
                    "user_prompt": None,
                    "raw_response": None,
                    "report": state.report,
                    "empty_report": True,
                },
                level="warning",
            )
            return WorkflowResult(
                success=True,
                content=state.report,
                metadata={
                    "research_id": state.id,
                    "finding_count": 0,
                    "empty_report": True,
                },
            )

        logger.info(
            "Starting synthesis phase: %d findings, %d sources",
            len(state.findings),
            len(state.sources),
        )

        # Build the synthesis prompt
        system_prompt = self._build_synthesis_system_prompt(state)
        user_prompt = self._build_synthesis_user_prompt(state)

        # Execute LLM call with context window error handling and timeout protection
        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id or state.synthesis_provider,
                model=state.synthesis_model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.5,  # Balanced for coherent but varied writing
                phase="synthesis",
                fallback_providers=self.config.get_phase_fallback_providers("synthesis"),
                max_retries=self.config.deep_research_max_retries,
                retry_delay=self.config.deep_research_retry_delay,
            )
        except ContextWindowError as e:
            logger.error(
                "Synthesis phase context window exceeded: prompt_tokens=%s, "
                "max_tokens=%s, truncation_needed=%s, provider=%s, finding_count=%d",
                e.prompt_tokens,
                e.max_tokens,
                e.truncation_needed,
                e.provider,
                len(state.findings),
            )
            return WorkflowResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "research_id": state.id,
                    "phase": "synthesis",
                    "error_type": "context_window_exceeded",
                    "prompt_tokens": e.prompt_tokens,
                    "max_tokens": e.max_tokens,
                    "truncation_needed": e.truncation_needed,
                    "finding_count": len(state.findings),
                    "guidance": "Try reducing the number of findings or source content included",
                },
            )

        if not result.success:
            # Check if this was a timeout
            if result.metadata and result.metadata.get("timeout"):
                logger.error(
                    "Synthesis phase timed out after exhausting all providers: %s",
                    result.metadata.get("providers_tried", []),
                )
            else:
                logger.error("Synthesis phase LLM call failed: %s", result.error)
            return result

        # Track token usage
        if result.tokens_used:
            state.total_tokens_used += result.tokens_used

        # Track phase metrics for audit
        state.phase_metrics.append(
            PhaseMetrics(
                phase="synthesis",
                duration_ms=result.duration_ms or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                provider_id=result.provider_id,
                model_used=result.model_used,
            )
        )

        # Extract the markdown report from the response
        report = self._extract_markdown_report(result.content)

        if not report:
            logger.warning("Failed to extract report from synthesis response")
            # Use raw content as fallback
            report = result.content

        # Store report in state
        state.report = report

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "synthesis_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "report": state.report,
                "report_length": len(state.report),
            },
        )

        logger.info(
            "Synthesis phase complete: report length %d chars",
            len(state.report),
        )

        return WorkflowResult(
            success=True,
            content=state.report,
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(state.findings),
                "source_count": len(state.sources),
                "report_length": len(state.report),
                "iteration": state.iteration,
            },
        )

    def _build_synthesis_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for report synthesis.

        Args:
            state: Current research state

        Returns:
            System prompt string
        """
        return """You are a research synthesizer. Your task is to create a comprehensive, well-structured research report from analyzed findings.

Generate a markdown-formatted report with the following structure:

# Research Report: [Topic]

## Executive Summary
A 2-3 paragraph overview of the key insights and conclusions.

## Key Findings

### [Theme/Category 1]
- Finding with supporting evidence and source citations [Source ID]
- Related findings grouped together

### [Theme/Category 2]
- Continue for each major theme...

## Analysis

### Supporting Evidence
Discussion of well-supported findings with high confidence.

### Conflicting Information
Note any contradictions or disagreements between sources (if present).

### Limitations
Acknowledge gaps in the research and areas needing further investigation.

## Sources
List sources as markdown links with their IDs: **[src-xxx]** [Title](URL)

## Conclusions
Actionable insights and recommendations based on the findings.

---

Guidelines:
- Organize findings thematically rather than listing them sequentially
- Cite source IDs in brackets when referencing specific information [src-xxx]
- Distinguish between high-confidence findings (well-supported) and lower-confidence insights
- Be specific and actionable in conclusions
- Keep the report focused on the original research query
- Use clear, professional language
- Include all relevant findings - don't omit information

IMPORTANT: Return ONLY the markdown report, no preamble or meta-commentary."""

    def _build_synthesis_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt with findings and sources for synthesis.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"# Research Query\n{state.original_query}",
            "",
            f"## Research Brief\n{state.research_brief or 'Direct research on the query'}",
            "",
            "## Findings to Synthesize",
            "",
        ]

        # Group findings by category if available
        categorized: dict[str, list] = {}
        uncategorized = []

        for finding in state.findings:
            category = finding.category or "General"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(finding)

        # Add findings by category
        for category, findings in categorized.items():
            prompt_parts.append(f"### {category}")
            for f in findings:
                confidence_label = f.confidence.value if hasattr(f.confidence, 'value') else str(f.confidence)
                source_refs = ", ".join(f.source_ids) if f.source_ids else "no sources"
                prompt_parts.append(f"- [{confidence_label.upper()}] {f.content}")
                prompt_parts.append(f"  Sources: {source_refs}")
            prompt_parts.append("")

        # Add knowledge gaps
        if state.gaps:
            prompt_parts.append("## Knowledge Gaps Identified")
            for gap in state.gaps:
                status = "addressed" if gap.resolved else "unresolved"
                prompt_parts.append(f"- [{status}] {gap.description}")
            prompt_parts.append("")

        # Add source reference list
        prompt_parts.append("## Source Reference")
        for source in state.sources[:30]:  # Limit to 30 for context window
            quality = source.quality.value if hasattr(source.quality, 'value') else str(source.quality)
            prompt_parts.append(f"- {source.id}: {source.title} [{quality}]")
            if source.url:
                prompt_parts.append(f"  URL: {source.url}")
        prompt_parts.append("")

        # Add synthesis instructions
        prompt_parts.extend([
            "## Instructions",
            f"Generate a comprehensive research report addressing the query: '{state.original_query}'",
            "",
            f"This is iteration {state.iteration} of {state.max_iterations}.",
            f"Total findings: {len(state.findings)}",
            f"Total sources: {len(state.sources)}",
            f"Unresolved gaps: {len(state.unresolved_gaps())}",
            "",
            "Create a well-structured markdown report following the format specified.",
        ])

        return "\n".join(prompt_parts)

    def _extract_markdown_report(self, content: str) -> Optional[str]:
        """Extract markdown report from LLM response.

        The response should be pure markdown, but this handles cases where
        the LLM wraps it in code blocks or adds preamble.

        Args:
            content: Raw LLM response content

        Returns:
            Extracted markdown report or None if extraction fails
        """
        if not content:
            return None

        # If content starts with markdown heading, it's likely clean
        if content.strip().startswith("#"):
            return content.strip()

        # Check for markdown code block wrapper
        if "```markdown" in content or "```md" in content:
            # Extract content between code blocks
            pattern = r'```(?:markdown|md)?\s*([\s\S]*?)```'
            matches = re.findall(pattern, content)
            if matches:
                return matches[0].strip()

        # Check for generic code block
        if "```" in content:
            pattern = r'```\s*([\s\S]*?)```'
            matches = re.findall(pattern, content)
            for match in matches:
                # Check if it looks like markdown (has headings)
                if match.strip().startswith("#") or "##" in match:
                    return match.strip()

        # Look for first heading and take everything from there
        heading_match = re.search(r'^(#[^\n]+)', content, re.MULTILINE)
        if heading_match:
            start_pos = heading_match.start()
            return content[start_pos:].strip()

        # If nothing else works, return the trimmed content
        return content.strip() if len(content.strip()) > 50 else None

    def _generate_empty_report(self, state: DeepResearchState) -> str:
        """Generate a minimal report when no findings are available.

        Args:
            state: Current research state

        Returns:
            Minimal markdown report
        """
        return f"""# Research Report

## Executive Summary

Research was conducted on the query: "{state.original_query}"

Unfortunately, the analysis phase did not yield extractable findings from the gathered sources. This may indicate:
- The sources lacked relevant information
- The query may need refinement
- Additional research iterations may be needed

## Research Query

{state.original_query}

## Research Brief

{state.research_brief or "No research brief generated."}

## Sources Examined

{len(state.sources)} source(s) were examined during this research session.

## Recommendations

1. Consider refining the research query for more specific results
2. Try additional research iterations if available
3. Review the gathered sources manually for relevant information

---

*Report generated with no extractable findings. Iteration {state.iteration}/{state.max_iterations}.*
"""

    async def _execute_refinement_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute refinement phase: analyze gaps and generate follow-up queries.

        This phase:
        1. Reviews the current report and identified gaps
        2. Uses LLM to assess gap severity and addressability
        3. Generates follow-up queries for unresolved gaps
        4. Converts high-priority gaps to new sub-queries for next iteration
        5. Respects max_iterations limit for workflow termination

        Args:
            state: Current research state with report and gaps
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with refinement outcome
        """
        unresolved_gaps = state.unresolved_gaps()

        # Check iteration limit
        if state.iteration >= state.max_iterations:
            logger.info(
                "Refinement: max iterations (%d) reached, no further refinement",
                state.max_iterations,
            )
            self._write_audit_event(
                state,
                "refinement_result",
                data={
                    "reason": "max_iterations_reached",
                    "unresolved_gaps": len(unresolved_gaps),
                    "iteration": state.iteration,
                },
                level="warning",
            )
            return WorkflowResult(
                success=True,
                content="Max iterations reached, refinement complete",
                metadata={
                    "research_id": state.id,
                    "iteration": state.iteration,
                    "max_iterations": state.max_iterations,
                    "unresolved_gaps": len(unresolved_gaps),
                    "reason": "max_iterations_reached",
                },
            )

        if not unresolved_gaps:
            logger.info("Refinement: no unresolved gaps, research complete")
            self._write_audit_event(
                state,
                "refinement_result",
                data={
                    "reason": "no_gaps",
                    "unresolved_gaps": 0,
                    "iteration": state.iteration,
                },
            )
            return WorkflowResult(
                success=True,
                content="No unresolved gaps, research complete",
                metadata={
                    "research_id": state.id,
                    "iteration": state.iteration,
                    "reason": "no_gaps",
                },
            )

        logger.info(
            "Starting refinement phase: %d unresolved gaps, iteration %d/%d",
            len(unresolved_gaps),
            state.iteration,
            state.max_iterations,
        )

        # Build the refinement prompt
        system_prompt = self._build_refinement_system_prompt(state)
        user_prompt = self._build_refinement_user_prompt(state)

        # Execute LLM call with context window error handling and timeout protection
        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id or state.refinement_provider,
                model=state.refinement_model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.4,  # Lower temperature for focused analysis
                phase="refinement",
                fallback_providers=self.config.get_phase_fallback_providers("refinement"),
                max_retries=self.config.deep_research_max_retries,
                retry_delay=self.config.deep_research_retry_delay,
            )
        except ContextWindowError as e:
            logger.error(
                "Refinement phase context window exceeded: prompt_tokens=%s, "
                "max_tokens=%s, gap_count=%d",
                e.prompt_tokens,
                e.max_tokens,
                len(unresolved_gaps),
            )
            return WorkflowResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "research_id": state.id,
                    "phase": "refinement",
                    "error_type": "context_window_exceeded",
                    "prompt_tokens": e.prompt_tokens,
                    "max_tokens": e.max_tokens,
                },
            )

        if not result.success:
            # Check if this was a timeout
            if result.metadata and result.metadata.get("timeout"):
                logger.error(
                    "Refinement phase timed out after exhausting all providers: %s",
                    result.metadata.get("providers_tried", []),
                )
            else:
                logger.error("Refinement phase LLM call failed: %s", result.error)
            return result

        # Track token usage
        if result.tokens_used:
            state.total_tokens_used += result.tokens_used

        # Track phase metrics for audit
        state.phase_metrics.append(
            PhaseMetrics(
                phase="refinement",
                duration_ms=result.duration_ms or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                provider_id=result.provider_id,
                model_used=result.model_used,
            )
        )

        # Parse the response
        parsed = self._parse_refinement_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse refinement response, using existing gap suggestions")
            # Fallback: use existing gap suggestions as follow-up queries
            follow_up_queries = self._extract_fallback_queries(state)
        else:
            follow_up_queries = parsed["follow_up_queries"]

            # Mark gaps as resolved if specified
            for gap_id in parsed.get("addressed_gap_ids", []):
                gap = state.get_gap(gap_id)
                if gap:
                    gap.resolved = True

        # Convert follow-up queries to new sub-queries for next iteration
        new_sub_queries = 0
        for query_data in follow_up_queries[:state.max_sub_queries]:
            # Add as new sub-query
            state.add_sub_query(
                query=query_data["query"],
                rationale=query_data.get("rationale", "Follow-up from gap analysis"),
                priority=query_data.get("priority", 1),
            )
            new_sub_queries += 1

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "refinement_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "parse_success": parsed["success"],
                "gap_analysis": parsed.get("gap_analysis", []),
                "follow_up_queries": follow_up_queries,
                "addressed_gap_ids": parsed.get("addressed_gap_ids", []),
                "should_iterate": parsed.get("should_iterate", True),
            },
        )

        logger.info(
            "Refinement phase complete: %d follow-up queries generated",
            new_sub_queries,
        )

        return WorkflowResult(
            success=True,
            content=f"Generated {new_sub_queries} follow-up queries from {len(unresolved_gaps)} gaps",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "iteration": state.iteration,
                "unresolved_gaps": len(unresolved_gaps),
                "follow_up_queries": new_sub_queries,
                "gaps_addressed": len(parsed.get("addressed_gap_ids", [])),
            },
        )

    def _build_refinement_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for gap analysis and refinement.

        Args:
            state: Current research state

        Returns:
            System prompt string
        """
        return """You are a research refiner. Your task is to analyze knowledge gaps identified during research and generate focused follow-up queries to address them.

Your response MUST be valid JSON with this exact structure:
{
    "gap_analysis": [
        {
            "gap_id": "gap-xxx",
            "severity": "critical|moderate|minor",
            "addressable": true,
            "rationale": "Why this gap matters and whether it can be addressed"
        }
    ],
    "follow_up_queries": [
        {
            "query": "A specific, focused search query to address the gap",
            "target_gap_id": "gap-xxx",
            "rationale": "How this query will fill the gap",
            "priority": 1
        }
    ],
    "addressed_gap_ids": ["gap-xxx"],
    "iteration_recommendation": {
        "should_iterate": true,
        "rationale": "Why iteration is or isn't recommended"
    }
}

Guidelines:
- Assess each gap's severity: "critical" (blocks conclusions), "moderate" (affects confidence), "minor" (nice to have)
- Only mark gaps as addressable if follow-up research can realistically fill them
- Generate 1-3 highly focused follow-up queries per addressable gap
- Priority 1 is highest priority
- Mark gaps as addressed if the current report already covers them adequately
- Recommend iteration only if there are addressable critical/moderate gaps AND value exceeds research cost

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_refinement_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt with gaps and report context for refinement.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"# Research Query\n{state.original_query}",
            "",
            f"## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Sources examined: {len(state.sources)}",
            f"- Findings extracted: {len(state.findings)}",
            f"- Unresolved gaps: {len(state.unresolved_gaps())}",
            "",
        ]

        # Add report summary (truncated for context window)
        if state.report:
            report_excerpt = state.report[:2000]
            if len(state.report) > 2000:
                report_excerpt += "\n\n[Report truncated...]"
            prompt_parts.append("## Current Report Summary")
            prompt_parts.append(report_excerpt)
            prompt_parts.append("")

        # Add unresolved gaps
        prompt_parts.append("## Unresolved Knowledge Gaps")
        for gap in state.unresolved_gaps():
            prompt_parts.append(f"\n### Gap: {gap.id}")
            prompt_parts.append(f"Description: {gap.description}")
            prompt_parts.append(f"Priority: {gap.priority}")
            if gap.suggested_queries:
                prompt_parts.append("Suggested queries from analysis:")
                for sq in gap.suggested_queries[:3]:
                    prompt_parts.append(f"  - {sq}")
        prompt_parts.append("")

        # Add high-confidence findings for context
        high_conf_findings = [
            f for f in state.findings
            if hasattr(f.confidence, 'value') and f.confidence.value in ('high', 'confirmed')
        ]
        if high_conf_findings:
            prompt_parts.append("## High-Confidence Findings Already Established")
            for f in high_conf_findings[:5]:
                prompt_parts.append(f"- {f.content[:200]}")
            prompt_parts.append("")

        # Add instructions
        prompt_parts.extend([
            "## Instructions",
            "1. Analyze each gap for severity and addressability",
            "2. Generate focused follow-up queries for addressable gaps",
            "3. Mark any gaps that are actually addressed by existing findings",
            "4. Recommend whether iteration is worthwhile given remaining gaps",
            "",
            "Return your analysis as JSON.",
        ])

        return "\n".join(prompt_parts)

    def _parse_refinement_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured refinement data.

        Args:
            content: Raw LLM response content
            state: Current research state

        Returns:
            Dict with 'success', 'follow_up_queries', 'addressed_gap_ids', etc.
        """
        result = {
            "success": False,
            "gap_analysis": [],
            "follow_up_queries": [],
            "addressed_gap_ids": [],
            "should_iterate": True,
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = self._extract_json(content)
        if not json_str:
            logger.warning("No JSON found in refinement response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from refinement response: %s", e)
            return result

        # Parse gap analysis
        raw_analysis = data.get("gap_analysis", [])
        if isinstance(raw_analysis, list):
            for ga in raw_analysis:
                if not isinstance(ga, dict):
                    continue
                result["gap_analysis"].append({
                    "gap_id": ga.get("gap_id", ""),
                    "severity": ga.get("severity", "moderate"),
                    "addressable": ga.get("addressable", True),
                    "rationale": ga.get("rationale", ""),
                })

        # Parse follow-up queries
        raw_queries = data.get("follow_up_queries", [])
        if isinstance(raw_queries, list):
            for fq in raw_queries:
                if not isinstance(fq, dict):
                    continue
                query = fq.get("query", "").strip()
                if not query:
                    continue
                result["follow_up_queries"].append({
                    "query": query,
                    "target_gap_id": fq.get("target_gap_id", ""),
                    "rationale": fq.get("rationale", ""),
                    "priority": min(max(int(fq.get("priority", 1)), 1), 10),
                })

        # Parse addressed gaps
        raw_addressed = data.get("addressed_gap_ids", [])
        if isinstance(raw_addressed, list):
            result["addressed_gap_ids"] = [
                gid for gid in raw_addressed if isinstance(gid, str)
            ]

        # Parse iteration recommendation
        iter_rec = data.get("iteration_recommendation", {})
        if isinstance(iter_rec, dict):
            result["should_iterate"] = iter_rec.get("should_iterate", True)

        # Mark success if we got at least one follow-up query
        result["success"] = len(result["follow_up_queries"]) > 0

        return result

    def _extract_fallback_queries(self, state: DeepResearchState) -> list[dict[str, Any]]:
        """Extract follow-up queries from existing gap suggestions as fallback.

        Used when LLM parsing fails but we still want to progress.

        Args:
            state: Current research state with gaps

        Returns:
            List of follow-up query dictionaries
        """
        queries = []
        for gap in state.unresolved_gaps():
            for i, sq in enumerate(gap.suggested_queries[:2]):  # Max 2 per gap
                queries.append({
                    "query": sq,
                    "target_gap_id": gap.id,
                    "rationale": f"Suggested query from gap: {gap.description[:50]}",
                    "priority": gap.priority,
                })
        return queries[:state.max_sub_queries]  # Respect limit

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_sessions(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        completed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List deep research sessions.

        Args:
            limit: Maximum sessions to return
            cursor: Pagination cursor (research_id to start after)
            completed_only: Only return completed sessions

        Returns:
            List of session summaries
        """
        sessions = self.memory.list_deep_research(
            limit=limit,
            cursor=cursor,
            completed_only=completed_only,
        )

        return [
            {
                "id": s.id,
                "query": s.original_query,
                "phase": s.phase.value,
                "iteration": s.iteration,
                "source_count": len(s.sources),
                "finding_count": len(s.findings),
                "is_complete": s.completed_at is not None,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]

    def delete_session(self, research_id: str) -> bool:
        """Delete a research session.

        Args:
            research_id: ID of session to delete

        Returns:
            True if deleted, False if not found
        """
        return self.memory.delete_deep_research(research_id)

    def resume_research(
        self,
        research_id: str,
        provider_id: Optional[str] = None,
        timeout_per_operation: float = 120.0,
        max_concurrent: int = 3,
    ) -> WorkflowResult:
        """Resume an interrupted deep research workflow from persisted state.

        Loads the DeepResearchState from persistence, validates it, and resumes
        execution from the current phase. Handles edge cases like corrupted
        state or missing sources gracefully.

        Args:
            research_id: ID of the research session to resume
            provider_id: Optional provider override for LLM operations
            timeout_per_operation: Timeout per operation in seconds
            max_concurrent: Maximum concurrent operations

        Returns:
            WorkflowResult with resumed research outcome or error
        """
        logger.info("Attempting to resume research session: %s", research_id)

        # Load existing state
        state = self.memory.load_deep_research(research_id)

        if state is None:
            logger.warning("Research session '%s' not found in persistence", research_id)
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found. It may have expired or been deleted.",
                metadata={"research_id": research_id, "error_type": "not_found"},
            )

        # Check if already completed
        if state.completed_at is not None:
            logger.info(
                "Research session '%s' already completed at %s",
                research_id,
                state.completed_at.isoformat(),
            )
            return WorkflowResult(
                success=True,
                content=state.report or "Research already completed",
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "is_complete": True,
                    "completed_at": state.completed_at.isoformat(),
                    "resumed": False,
                },
            )

        # Validate state integrity
        validation_result = self._validate_state_for_resume(state)
        if not validation_result["valid"]:
            logger.error(
                "Research session '%s' failed validation: %s",
                research_id,
                validation_result["error"],
            )
            return WorkflowResult(
                success=False,
                content="",
                error=validation_result["error"],
                metadata={
                    "research_id": research_id,
                    "error_type": "validation_failed",
                    "phase": state.phase.value,
                    "issues": validation_result.get("issues", []),
                },
            )

        # Log resumption context
        logger.info(
            "Resuming research '%s': phase=%s, iteration=%d/%d, "
            "sub_queries=%d (completed=%d), sources=%d, findings=%d, gaps=%d",
            research_id,
            state.phase.value,
            state.iteration,
            state.max_iterations,
            len(state.sub_queries),
            len(state.completed_sub_queries()),
            len(state.sources),
            len(state.findings),
            len(state.unresolved_gaps()),
        )

        # Resume workflow execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        ),
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                )
        except RuntimeError:
            result = asyncio.run(
                self._execute_workflow_async(
                    state=state,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
            )

        # Add resumption metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata["resumed"] = True
        result.metadata["resumed_from_phase"] = state.phase.value

        return result

    def _validate_state_for_resume(self, state: DeepResearchState) -> dict[str, Any]:
        """Validate a DeepResearchState for safe resumption.

        Checks for common corruption issues and missing required data.

        Args:
            state: The state to validate

        Returns:
            Dict with 'valid' bool and 'error'/'issues' if invalid
        """
        issues = []

        # Check required fields
        if not state.original_query:
            issues.append("Missing original_query")

        if not state.id:
            issues.append("Missing research ID")

        # Phase-specific validation
        if state.phase.value in ("gathering", "analysis", "synthesis", "refinement"):
            # These phases require sub-queries from planning
            if not state.sub_queries:
                issues.append(f"No sub-queries found for {state.phase.value} phase")

        if state.phase.value in ("analysis", "synthesis"):
            # These phases require sources from gathering
            if not state.sources and state.phase.value == "analysis":
                # Only warn for analysis - synthesis can work with findings
                issues.append("No sources found for analysis phase")

        if state.phase.value == "synthesis":
            # Synthesis requires findings from analysis
            if not state.findings:
                issues.append("No findings found for synthesis phase")

        # Check for corrupted collections (None instead of empty list)
        if state.sub_queries is None:
            issues.append("Corrupted sub_queries collection (null)")
        if state.sources is None:
            issues.append("Corrupted sources collection (null)")
        if state.findings is None:
            issues.append("Corrupted findings collection (null)")
        if state.gaps is None:
            issues.append("Corrupted gaps collection (null)")

        if issues:
            return {
                "valid": False,
                "error": f"State validation failed: {'; '.join(issues)}",
                "issues": issues,
            }

        return {"valid": True}

    def list_resumable_sessions(self) -> list[dict[str, Any]]:
        """List all in-progress research sessions that can be resumed.

        Scans persistence for sessions that are not completed and can be resumed.

        Returns:
            List of session summaries with resumption context
        """
        sessions = self.memory.list_deep_research(completed_only=False)

        resumable = []
        for state in sessions:
            if state.completed_at is not None:
                continue  # Skip completed

            validation = self._validate_state_for_resume(state)

            resumable.append({
                "id": state.id,
                "query": state.original_query[:100] + ("..." if len(state.original_query) > 100 else ""),
                "phase": state.phase.value,
                "iteration": state.iteration,
                "max_iterations": state.max_iterations,
                "sub_queries": len(state.sub_queries),
                "completed_queries": len(state.completed_sub_queries()),
                "sources": len(state.sources),
                "findings": len(state.findings),
                "gaps": len(state.unresolved_gaps()),
                "can_resume": validation["valid"],
                "issues": validation.get("issues", []),
                "created_at": state.created_at.isoformat(),
                "updated_at": state.updated_at.isoformat(),
            })

        return resumable
