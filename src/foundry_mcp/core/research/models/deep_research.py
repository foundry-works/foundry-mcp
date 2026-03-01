"""Deep research workflow models (multi-phase iterative research)."""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

from foundry_mcp.core.research.models.digest import make_fragment_id, parse_fragment_id
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.fidelity import (
    ContentFidelityRecord,
    FidelityLevel,
    PhaseMetrics,
)
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchGap,
    ResearchMode,
    ResearchSource,
    SourceType,
    SubQuery,
)

# Single source of truth for the default supervision rounds cap.
DEFAULT_MAX_SUPERVISION_ROUNDS: int = 6

# Bounded state growth caps — prevent unbounded memory accumulation in
# long-running research sessions (see PLAN.md Phase 2).
MAX_SOURCES: int = 500
MAX_TOPIC_RESEARCH_RESULTS: int = 50
MAX_SUPERVISION_MESSAGES: int = 100


class TopicResearchResult(BaseModel):
    """Result of a per-topic ReAct research loop.

    Each sub-query can be investigated independently by a topic researcher
    that runs its own search → reflect → refine cycle. This model captures
    the outcome of that per-topic investigation.
    """

    sub_query_id: str = Field(..., description="ID of the SubQuery this result belongs to")
    searches_performed: int = Field(default=0, description="Number of search iterations executed")
    sources_found: int = Field(default=0, description="Total unique sources discovered for this topic")
    per_topic_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated summary of findings for this specific topic",
    )
    reflection_notes: list[str] = Field(
        default_factory=list,
        description="Notes from per-topic reflection steps (e.g., identified gaps, query refinements)",
    )
    refined_queries: list[str] = Field(
        default_factory=list,
        description="Refined queries generated during the ReAct loop",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of sources discovered by this topic researcher",
    )
    message_history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Raw message history from the ReAct research loop. Contains "
            "all assistant responses, tool results, and system messages in "
            "chronological order. Passed to compression for higher-quality "
            "citation-rich summaries that preserve the researcher's reasoning chain."
        ),
    )
    raw_notes: Optional[str] = Field(
        default=None,
        description=(
            "Unprocessed concatenation of all tool-result and assistant messages "
            "from the ReAct research loop. Captured before compression so that "
            "raw evidence survives even if compression fails or drops content. "
            "Used by synthesis as supplementary context and by the groundedness "
            "evaluator as ground-truth evidence."
        ),
    )
    compressed_findings: Optional[str] = Field(
        default=None,
        description=(
            "Citation-rich compressed summary of this topic's sources, "
            "produced by the per-topic compression step before analysis. "
            "When present, analysis prefers this over raw source content."
        ),
    )
    supervisor_summary: Optional[str] = Field(
        default=None,
        description=(
            "Short, structured summary optimized for supervisor gap analysis. "
            "Highlights what was covered, key findings, and remaining uncertainty. "
            "Generated alongside compressed_findings but with different objectives."
        ),
    )
    tool_parse_failures: int = Field(
        default=0,
        description=(
            "Number of times the researcher LLM returned a response that could "
            "not be parsed as valid tool-call JSON. Retried with a clarifying "
            "suffix up to 2 times per turn. High values indicate model/prompt "
            "format issues."
        ),
    )
    compression_messages_dropped: int = Field(
        default=0,
        description=(
            "Number of messages dropped from the ReAct message history during "
            "compression retries due to token-limit errors. Higher values indicate "
            "the compression model's context window was significantly exceeded."
        ),
    )
    compression_retry_count: int = Field(
        default=0,
        description=(
            "Number of compression retries needed due to token-limit errors. "
            "Each retry drops the oldest messages from the history while "
            "preserving the most recent think messages and search results."
        ),
    )
    compression_original_message_count: int = Field(
        default=0,
        description=(
            "Original number of messages in the ReAct history before any "
            "compression truncation. Zero when message_history was empty "
            "(structured metadata fallback path)."
        ),
    )
    early_completion: bool = Field(
        default=False,
        description="Whether the topic researcher signalled early research completion",
    )
    completion_rationale: str = Field(
        default="",
        description="Rationale provided by the reflection step for early completion or exit",
    )


class ResearchDirective(BaseModel):
    """A structured research delegation directive from the supervisor.

    When the supervisor identifies gaps in research coverage, it generates
    detailed directives that are executed by parallel topic researchers.
    Each directive is a paragraph-length description of what to investigate,
    what perspective to take, and what evidence to seek — more targeted than
    a simple follow-up query string.

    Analogous to open_deep_research's ``ConductResearch`` tool output.
    """

    id: str = Field(default_factory=lambda: f"dir-{uuid4().hex[:8]}")
    research_topic: str = Field(
        ...,
        description=(
            "Detailed paragraph-length directive describing what to investigate. "
            "Should specify the research approach (compare, investigate, validate), "
            "target specific gaps, and describe expected deliverables."
        ),
    )
    perspective: str = Field(
        default="",
        description="What angle or perspective to approach the research from",
    )
    evidence_needed: str = Field(
        default="",
        description="What specific evidence types or data points to seek",
    )
    priority: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Priority: 1=critical, 2=important, 3=nice-to-have",
    )
    supervision_round: int = Field(
        default=0,
        description="The supervision round that generated this directive",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =========================================================================
# Structured output schemas for LLM boundaries (Phase 4 PLAN)
# =========================================================================


class DelegationResponse(BaseModel):
    """Structured output schema for supervision delegation LLM calls.

    Replaces free-form JSON parsing of delegation responses. The supervisor
    LLM produces this schema to indicate whether research is complete or
    to generate new research directives.

    Used with ``execute_structured_llm_call()`` for type-safe parsing with
    automatic retry on validation failure.
    """

    research_complete: bool = Field(
        default=False,
        description="True when existing coverage is sufficient across all dimensions",
    )
    directives: list[ResearchDirective] = Field(
        default_factory=list,
        description="Research directives targeting identified gaps",
    )
    rationale: str = Field(
        default="",
        description="Why these directives were chosen or why research is complete",
    )

    @model_validator(mode="after")
    def _fix_incomplete_without_directives(self) -> "DelegationResponse":
        """If not research_complete and no directives, force research_complete=True."""
        if not self.research_complete and not self.directives:
            logger.warning(
                "DelegationResponse has research_complete=False with empty directives; forcing research_complete=True"
            )
            self.research_complete = True
        return self


class ReflectionDecision(BaseModel):
    """Structured output schema for topic researcher reflection LLM calls.

    Replaces free-form JSON parsing of reflection decisions. The researcher
    LLM produces this schema after each search iteration to decide next steps.

    Used with ``execute_structured_llm_call()`` for type-safe parsing with
    automatic retry on validation failure.
    """

    continue_searching: bool = Field(
        default=False,
        description="Whether to continue searching for more sources",
    )
    research_complete: bool = Field(
        default=False,
        description=(
            "True when the researcher is confident findings address the "
            "research question. Stronger signal than continue_searching=False."
        ),
    )
    refined_query: Optional[str] = Field(
        default=None,
        description="Refined search query if continuing (null if stopping)",
    )
    urls_to_extract: list[str] = Field(
        default_factory=list,
        description="URLs to extract full content from (max 2 recommended)",
    )
    rationale: str = Field(
        default="",
        description="Explanation of the assessment and what gap remains if continuing",
    )

    @field_validator("urls_to_extract", mode="before")
    @classmethod
    def _coerce_urls(cls, v: Any) -> list[str]:
        """Accept null/None as empty list, filter to valid HTTP URLs with SSRF protection.

        Uses ``resolve_dns=True`` because these URLs originate from LLM output
        and will be used for server-side fetch — DNS rebinding must be blocked.
        """
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            validate_extract_url,
        )

        if v is None:
            return []
        if not isinstance(v, list):
            return []
        return [str(u).strip() for u in v if isinstance(u, str) and validate_extract_url(u.strip(), resolve_dns=True)][
            :5
        ]  # hard cap for safety

    @model_validator(mode="after")
    def _fix_contradictory_flags(self) -> "ReflectionDecision":
        """If research_complete is True, force continue_searching to False."""
        if self.research_complete and self.continue_searching:
            logger.warning(
                "ReflectionDecision has both research_complete=True and "
                "continue_searching=True; forcing continue_searching=False"
            )
            self.continue_searching = False
        return self


class ResearchBriefOutput(BaseModel):
    """Structured output schema for research brief generation LLM calls.

    Replaces raw text parsing of brief output. The brief LLM produces this
    schema to provide a structured research brief with optional scope and
    source preference metadata.

    Used with ``execute_structured_llm_call()`` for type-safe parsing with
    automatic retry on validation failure.
    """

    research_brief: str = Field(
        ...,
        description="The detailed, structured research brief paragraph(s)",
    )
    scope_boundaries: Optional[str] = Field(
        default=None,
        description="What the research should include and exclude",
    )
    source_preferences: Optional[str] = Field(
        default=None,
        description="Preferred source types (primary, official, peer-reviewed, etc.)",
    )


# =========================================================================
# Parse functions for execute_structured_llm_call()
# =========================================================================


def parse_delegation_response(content: str) -> DelegationResponse:
    """Parse raw LLM content into a validated DelegationResponse.

    Extracts JSON from the LLM response (handling markdown code blocks
    and surrounding text), then validates against the DelegationResponse
    schema. Raises ValueError on extraction or validation failure.

    Intended as the ``parse_fn`` argument to ``execute_structured_llm_call()``.

    Args:
        content: Raw LLM response text

    Returns:
        Validated DelegationResponse instance

    Raises:
        ValueError: If no JSON found or validation fails
    """
    import json as _json

    from foundry_mcp.core.research.workflows.deep_research._helpers import (
        extract_json,
    )

    json_str = extract_json(content)
    if not json_str:
        raise ValueError("No JSON object found in delegation response")

    data = _json.loads(json_str)
    return DelegationResponse.model_validate(data)


def parse_reflection_decision(content: str) -> ReflectionDecision:
    """Parse raw LLM content into a validated ReflectionDecision.

    Extracts JSON from the LLM response, then validates against the
    ReflectionDecision schema. Raises ValueError on extraction or
    validation failure.

    Intended as the ``parse_fn`` argument to ``execute_structured_llm_call()``.

    Args:
        content: Raw LLM response text

    Returns:
        Validated ReflectionDecision instance

    Raises:
        ValueError: If no JSON found or validation fails
    """
    import json as _json

    from foundry_mcp.core.research.workflows.deep_research._helpers import (
        extract_json,
    )

    json_str = extract_json(content)
    if not json_str:
        raise ValueError("No JSON object found in reflection response")

    data = _json.loads(json_str)
    return ReflectionDecision.model_validate(data)


def parse_brief_output(content: str) -> ResearchBriefOutput:
    """Parse raw LLM content into a validated ResearchBriefOutput.

    Attempts JSON extraction first. If the response contains valid JSON
    matching the schema, returns the parsed model. If no JSON is found
    but the content is non-empty, treats the entire content as the
    ``research_brief`` field (backward compatibility with plain-text briefs).

    Intended as the ``parse_fn`` argument to ``execute_structured_llm_call()``.

    Args:
        content: Raw LLM response text

    Returns:
        Validated ResearchBriefOutput instance

    Raises:
        ValueError: If content is empty or validation fails
    """
    import json as _json

    from foundry_mcp.core.research.workflows.deep_research._helpers import (
        extract_json,
    )

    if not content or not content.strip():
        raise ValueError("Empty brief response")

    # Try JSON extraction first
    json_str = extract_json(content)
    if json_str:
        try:
            data = _json.loads(json_str)
            return ResearchBriefOutput.model_validate(data)
        except (_json.JSONDecodeError, ValueError):
            pass  # Fall through to plain-text handling

    # Backward compat: treat plain text as the research_brief field
    return ResearchBriefOutput(research_brief=content.strip())


# =========================================================================
# Researcher tool schemas (Phase 2 PLAN: Tool-Calling Researchers)
# =========================================================================


class WebSearchTool(BaseModel):
    """Tool schema for web search.

    The researcher calls this to search the web for information on a topic.
    Dispatched to configured search providers (Tavily, Perplexity, etc.).

    Supports both single-query (``query``) and batch (``queries``) forms.
    The model_validator normalizes both into ``queries`` so the handler
    always works with a list.
    """

    query: str | None = Field(default=None, description="Single search query string")
    queries: list[str] | None = Field(
        default=None,
        description="Batch of search queries to execute in parallel",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return per query",
    )

    _MAX_QUERIES: ClassVar[int] = 10

    @model_validator(mode="after")
    def _normalize_queries(self) -> "WebSearchTool":
        """Normalize single query and batch queries into ``self.queries``."""
        if self.queries and self.query:
            # Both provided — merge single into batch if not already present
            if self.query not in self.queries:
                self.queries = [self.query] + list(self.queries)
        elif self.query:
            self.queries = [self.query]
        elif not self.queries:
            raise ValueError("Either 'query' or 'queries' must be provided")
        # Cap the number of queries to prevent unbounded search API dispatch.
        if len(self.queries) > self._MAX_QUERIES:
            logger.warning(
                "WebSearchTool received %d queries; truncating to %d.",
                len(self.queries),
                self._MAX_QUERIES,
            )
            self.queries = self.queries[: self._MAX_QUERIES]
        # Ensure query reflects first entry for backward compat
        self.query = self.queries[0]
        return self


class ExtractContentTool(BaseModel):
    """Tool schema for extracting full page content from URLs.

    The researcher calls this when a search result snippet suggests
    rich content that warrants full extraction (technical docs, papers, etc.).
    """

    urls: list[str] = Field(
        ...,
        description="URLs to extract full content from (max 2 per call)",
    )

    @field_validator("urls", mode="before")
    @classmethod
    def _cap_urls(cls, v: Any) -> list[str]:
        """Cap at 2 URLs per extraction call with SSRF validation.

        Uses ``resolve_dns=False`` to avoid blocking DNS lookups inside a
        synchronous Pydantic validator.  IP-literal SSRF attacks are caught
        here; DNS-rebinding attacks are caught downstream by the async
        extraction layer (``validate_extract_url_async``).
        """
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            validate_extract_url,
        )

        if not isinstance(v, list):
            return []
        safe: list[str] = []
        for u in v:
            if not isinstance(u, str):
                continue
            stripped = u.strip()
            try:
                if validate_extract_url(stripped, resolve_dns=False):
                    safe.append(stripped)
            except Exception:
                continue
        return safe[:2]


class ThinkTool(BaseModel):
    """Tool schema for strategic reflection.

    The researcher calls this to pause and assess findings, identify gaps,
    and plan next steps. Does NOT count against the tool call budget.
    """

    reasoning: str = Field(
        ...,
        description="Strategic reasoning about research progress, gaps, and next steps",
    )


class ResearchCompleteTool(BaseModel):
    """Tool schema for signaling research completion.

    The researcher calls this when confident that findings adequately
    address the research question. Terminates the ReAct loop.
    """

    summary: str = Field(
        ...,
        description="Summary of findings that address the research question",
    )


class ResearcherToolCall(BaseModel):
    """A single tool call from the researcher LLM.

    The researcher responds with a list of these to indicate which
    tools to execute in a given turn.
    """

    tool: str = Field(
        ...,
        description="Tool name: web_search, extract_content, think, research_complete",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments matching the tool's schema",
    )


class ResearcherResponse(BaseModel):
    """Structured response from the researcher LLM in a ReAct turn.

    The researcher responds with tool calls indicating what actions to take.
    Optionally includes brief reasoning before the tool calls.
    """

    tool_calls: list[ResearcherToolCall] = Field(
        default_factory=list,
        description="Tools to execute this turn",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional brief reasoning before tool calls",
    )
    parse_failed: bool = Field(
        default=False,
        description=(
            "True when the LLM returned non-empty content that could not be "
            "parsed as valid tool-call JSON. Used by the ReAct loop to trigger "
            "retry with a clarifying suffix instead of stopping."
        ),
    )


#: Registry mapping tool names to their Pydantic schema classes.
RESEARCHER_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "web_search": WebSearchTool,
    "extract_content": ExtractContentTool,
    "think": ThinkTool,
    "research_complete": ResearchCompleteTool,
}

#: Tools that do NOT count against the researcher's budget.
BUDGET_EXEMPT_TOOLS: frozenset[str] = frozenset({"think", "research_complete"})


def parse_researcher_response(content: str) -> ResearcherResponse:
    """Parse raw LLM content into a validated ResearcherResponse.

    Extracts JSON from the LLM response (handling markdown code blocks
    and surrounding text), then validates against the ResearcherResponse
    schema.

    On parse failure of non-empty content, returns a response with
    ``parse_failed=True`` so the caller can retry with a clarifying suffix.
    Empty/blank content returns an empty response (graceful stop, no retry).

    Args:
        content: Raw LLM response text

    Returns:
        Validated ResearcherResponse instance. Check ``parse_failed`` to
        distinguish parse errors (retryable) from genuine empty responses.
    """
    import json as _json

    from foundry_mcp.core.research.workflows.deep_research._helpers import (
        extract_json,
    )

    if not content or not content.strip():
        return ResearcherResponse()

    json_str = extract_json(content)
    if not json_str:
        return ResearcherResponse(parse_failed=True)

    try:
        data = _json.loads(json_str)
        return ResearcherResponse.model_validate(data)
    except (_json.JSONDecodeError, ValueError, TypeError):
        return ResearcherResponse(parse_failed=True)


class Contradiction(BaseModel):
    """A contradiction detected between research findings.

    Identified during the analysis phase when multiple sources provide
    conflicting information on the same topic. Contradictions are surfaced
    in the synthesis prompt so the final report can address them explicitly.
    """

    id: str = Field(default_factory=lambda: f"contra-{uuid4().hex[:8]}")
    finding_ids: list[str] = Field(
        ...,
        description="IDs of the conflicting ResearchFinding objects",
    )
    description: str = Field(
        ...,
        description="Description of the conflict between findings",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Suggested resolution or explanation for the contradiction",
    )
    preferred_source_id: Optional[str] = Field(
        default=None,
        description="ID of the more authoritative source, if determinable",
    )
    severity: Literal["major", "minor"] = Field(
        default="minor",
        description="Severity of the contradiction: 'major' or 'minor'",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeepResearchConfig(BaseModel):
    """Configuration for DEEP_RESEARCH workflow execution.

    Groups deep research parameters into a single config object to reduce
    parameter sprawl in the MCP tool interface. All fields have sensible
    defaults that can be overridden at the tool level.

    Note: Provider configuration is handled via ResearchConfig TOML settings,
    not through this config object. This is intentional - providers should be
    configured at the server level, not per-request.
    """

    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum refinement iterations before forced completion",
    )
    max_sub_queries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum sub-queries for query decomposition",
    )
    max_sources_per_query: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum sources to gather per sub-query",
    )
    follow_links: bool = Field(
        default=True,
        description="Whether to follow URLs and extract full content",
    )
    timeout_per_operation: float = Field(
        default=360.0,
        ge=1.0,
        le=1800.0,
        description="Timeout in seconds for each search/fetch operation",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent operations (search, fetch)",
    )

    @classmethod
    def from_defaults(cls) -> "DeepResearchConfig":
        """Create config with all default values.

        Returns:
            DeepResearchConfig with sensible defaults
        """
        return cls()

    def merge_overrides(self, **overrides: Any) -> "DeepResearchConfig":
        """Create a new config with specified overrides applied.

        Args:
            **overrides: Field values to override (None values are ignored)

        Returns:
            New DeepResearchConfig with overrides applied
        """
        current = self.model_dump()
        for key, value in overrides.items():
            if value is not None and key in current:
                current[key] = value
        return DeepResearchConfig(**current)


class DeepResearchPhase(str, Enum):
    """Phases of the DEEP_RESEARCH workflow.

    Active workflow progression:
    0. CLARIFICATION - (Optional) Analyze query specificity and ask clarifying questions
    1. BRIEF - Enrich the raw query into a structured research brief
    2. SUPERVISION - Supervisor-owned decomposition (round 0) and gap-fill (rounds 1+)
    3. SYNTHESIS - Combine findings into a comprehensive report

    PLANNING and GATHERING are retained only for legacy saved-state resume
    compatibility.  New workflows proceed directly from
    BRIEF → SUPERVISION → SYNTHESIS.

    The ordering of these enum values is significant - it defines the
    progression through advance_phase() method.
    """

    CLARIFICATION = "clarification"
    BRIEF = "brief"
    PLANNING = "planning"  # DEPRECATED: legacy-resume-only; retained for deserialization
    GATHERING = "gathering"  # DEPRECATED: legacy-resume-only; new workflows skip to SUPERVISION
    ANALYSIS = "analysis"  # DEPRECATED: legacy-resume-only; retained for deserialization
    REFINEMENT = "refinement"  # DEPRECATED: legacy-resume-only; retained for deserialization
    SUPERVISION = "supervision"
    SYNTHESIS = "synthesis"


class ResearchExtensions(BaseModel):
    """Container for extended research capabilities.

    All fields from PLAN-1 through PLAN-4 live here rather than
    directly on DeepResearchState. This keeps the core state model
    stable and serialization cost proportional to features used.

    Fields are populated lazily by each plan's implementation:
    - PLAN-1: research_profile, provenance, structured_output
    - PLAN-3: research_landscape
    - PLAN-4: citation_network, methodology_assessments
    """

    # PLAN-1: Foundations
    research_profile: Optional[Any] = Field(
        default=None,
        description="Research profile from PLAN-1 (forward reference placeholder)",
    )
    provenance: Optional[Any] = Field(
        default=None,
        description="Provenance log from PLAN-1 (forward reference placeholder)",
    )
    structured_output: Optional[Any] = Field(
        default=None,
        description="Structured research output from PLAN-1 (forward reference placeholder)",
    )

    # PLAN-3: Intelligence
    research_landscape: Optional[Any] = Field(
        default=None,
        description="Research landscape from PLAN-3 (forward reference placeholder)",
    )

    # PLAN-4: Deep Analysis
    citation_network: Optional[Any] = Field(
        default=None,
        description="Citation network from PLAN-4 (forward reference placeholder)",
    )
    methodology_assessments: list[Any] = Field(
        default_factory=list,
        description="Methodology assessments from PLAN-4 (forward reference placeholder)",
    )

    model_config = {"extra": "forbid"}

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to exclude None fields by default."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class DeepResearchState(BaseModel):
    """Main state model for a deep research session.

    Manages the entire lifecycle of a multi-phase research workflow:
    - Tracks the current phase and iteration
    - Contains all sub-queries, sources, findings, and gaps
    - Provides helper methods for state manipulation
    - Handles phase advancement and refinement iteration logic

    The state is persisted to enable session resume capability.
    """

    id: str = Field(default_factory=lambda: f"deepres-{uuid4().hex[:12]}")
    original_query: str = Field(..., description="The original research query")
    clarification_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints and context inferred or provided during CLARIFICATION phase",
    )
    research_brief: Optional[str] = Field(
        default=None,
        description="Enriched research brief generated in BRIEF phase (or PLANNING fallback)",
    )
    phase: DeepResearchPhase = Field(
        default=DeepResearchPhase.CLARIFICATION,
        description="Current workflow phase",
    )
    iteration: int = Field(
        default=1,
        description="Current refinement iteration (1-based)",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum refinement iterations before forced completion",
    )

    # Collections
    sub_queries: list[SubQuery] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Contradictions detected between findings during analysis",
    )
    topic_research_results: list[TopicResearchResult] = Field(
        default_factory=list,
        description="Per-topic research results from parallel topic researcher agents",
    )

    # Supervisor delegation directives (Phase 4 PLAN)
    directives: list[ResearchDirective] = Field(
        default_factory=list,
        description=(
            "Research directives generated by the supervisor delegation model. "
            "Each directive is a paragraph-length research assignment executed "
            "by a parallel topic researcher. Tracked for audit traceability."
        ),
    )

    # Session-level aggregated raw notes from all topic researchers
    raw_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Aggregated raw notes from all topic researchers (gathering and "
            "supervision phases). Each entry is the unprocessed concatenation "
            "of tool-result and assistant messages from one researcher's ReAct "
            "loop. Used by synthesis as supplementary context when token budget "
            "allows, and by the groundedness evaluator as ground-truth evidence."
        ),
    )

    # Global compression output (cross-topic deduplication digest)
    compressed_digest: Optional[str] = Field(
        default=None,
        description=(
            "Unified research digest produced by global compression. "
            "Deduplicates cross-topic findings, merges themes, and "
            "flags contradictions. When present, synthesis prefers this "
            "over raw findings for prompt construction."
        ),
    )

    # Final output
    report: Optional[str] = Field(
        default=None,
        description="Final synthesized research report",
    )
    report_output_path: Optional[str] = Field(
        default=None,
        description="Path to saved markdown report file",
    )
    report_sections: dict[str, str] = Field(
        default_factory=dict,
        description="Named sections of the report for structured access",
    )

    # Execution tracking
    total_sources_examined: int = Field(default=0)
    total_tokens_used: int = Field(default=0)
    total_duration_ms: float = Field(default=0.0)

    # Per-phase metrics for audit
    phase_metrics: list[PhaseMetrics] = Field(
        default_factory=list,
        description="Metrics for each executed phase (timing, tokens, provider)",
    )
    # Search provider query counts (provider_name -> query_count)
    search_provider_stats: dict[str, int] = Field(
        default_factory=dict,
        description="Count of queries executed per search provider",
    )

    # Polling tracking
    status_check_count: int = Field(
        default=0,
        description="Number of status checks made",
    )
    last_status_check_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last status check",
    )

    # Heartbeat tracking for progress visibility
    last_heartbeat_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last heartbeat (updated before provider calls)",
    )

    # Content fidelity tracking (for token budget management)
    # Per-item fidelity records: content_fidelity[item_id].phases[phase] = {level, reason, warnings, timestamp}
    content_fidelity: dict[str, ContentFidelityRecord] = Field(
        default_factory=dict,
        description="Per-item fidelity records tracking degradation across phases",
    )
    dropped_content_ids: list[str] = Field(
        default_factory=list,
        description="IDs of sources dropped during budget allocation",
    )
    content_allocation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate metadata: total_tokens_used, overall_fidelity_score, phase_budgets, warnings",
    )

    # Configuration
    source_types: list[SourceType] = Field(
        default_factory=lambda: [SourceType.WEB, SourceType.ACADEMIC],
    )
    max_sources_per_query: int = Field(default=5)
    max_sub_queries: int = Field(default=5)
    follow_links: bool = Field(
        default=True,
        description="Whether to follow URLs and extract full content",
    )
    research_mode: ResearchMode = Field(
        default=ResearchMode.GENERAL,
        description="Research mode for source prioritization",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)

    # Provider tracking (per-phase LLM provider configuration)
    # Supports ProviderSpec format: "[cli]gemini:pro" or simple names: "gemini"
    planning_provider: Optional[str] = Field(default=None)
    analysis_provider: Optional[str] = Field(default=None)
    synthesis_provider: Optional[str] = Field(default=None)
    refinement_provider: Optional[str] = Field(default=None)
    # Per-phase model overrides (from ProviderSpec parsing)
    planning_model: Optional[str] = Field(default=None)
    analysis_model: Optional[str] = Field(default=None)
    synthesis_model: Optional[str] = Field(default=None)
    refinement_model: Optional[str] = Field(default=None)

    # Supervision tracking (iterative coverage assessment)
    supervision_round: int = Field(
        default=0,
        description="Current supervision round within this iteration (0-based, resets each refinement iteration)",
    )
    max_supervision_rounds: int = Field(
        default=DEFAULT_MAX_SUPERVISION_ROUNDS,
        description="Maximum supervisor assess-delegate rounds per iteration",
    )
    supervision_provider: Optional[str] = Field(default=None)
    supervision_model: Optional[str] = Field(default=None)
    supervision_messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Accumulated supervisor conversation across delegation rounds. "
            "Contains the supervisor's prior think outputs, delegation responses, "
            "and compressed research findings from each executed directive. "
            "Passed to the delegation LLM on subsequent rounds so it can "
            "reference its own prior reasoning and the full research context."
        ),
    )

    system_prompt: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Extended capabilities container (PLAN-1 through PLAN-4).
    # Uses default_factory so the field is always present but lightweight
    # when unused — exclude_none in ResearchExtensions.model_dump() means
    # empty extensions add zero overhead to state serialization.
    extensions: ResearchExtensions = Field(
        default_factory=ResearchExtensions,
        description="Extended capabilities from PLAN-1 through PLAN-4",
    )

    # Citation counter — maintained by add_source()/append_source().
    # Avoids O(n) scan of all sources on every add.
    next_citation_number: int = Field(
        default=1,
        description="Next citation number to assign (auto-maintained).",
    )

    # Deprecated phase values — log a warning when loaded from persisted state.
    _DEPRECATED_PHASES: ClassVar[set[DeepResearchPhase]] = {
        DeepResearchPhase.PLANNING,
        DeepResearchPhase.GATHERING,
        DeepResearchPhase.ANALYSIS,
        DeepResearchPhase.REFINEMENT,
    }

    @field_validator("phase", mode="after")
    @classmethod
    def _warn_deprecated_phase(cls, v: DeepResearchPhase) -> DeepResearchPhase:
        """Log a deprecation warning when loading a session with a removed phase."""
        if v in cls._DEPRECATED_PHASES:
            logger.warning(
                "DeepResearchState loaded with deprecated phase %r; advance_phase() will skip past it automatically.",
                v.value,
            )
        return v

    @model_validator(mode="after")
    def _sync_citation_counter(self) -> "DeepResearchState":
        """Ensure citation counter is consistent with existing sources.

        Handles backward compatibility when deserializing sessions saved
        before this field existed (default=1 is bumped up to match sources).
        """
        if self.sources:
            max_existing = max((s.citation_number or 0 for s in self.sources), default=0)
            if max_existing >= self.next_citation_number:
                self.next_citation_number = max_existing + 1
        return self

    # =========================================================================
    # Extension convenience accessors
    # =========================================================================

    @property
    def research_profile(self) -> Optional[Any]:
        """Convenience accessor for extensions.research_profile."""
        return self.extensions.research_profile

    @property
    def provenance(self) -> Optional[Any]:
        """Convenience accessor for extensions.provenance."""
        return self.extensions.provenance

    # =========================================================================
    # Collection Management Methods
    # =========================================================================

    def add_sub_query(
        self,
        query: str,
        rationale: Optional[str] = None,
        priority: int = 1,
    ) -> SubQuery:
        """Add a new sub-query for research.

        Args:
            query: The focused sub-query text
            rationale: Why this sub-query was generated
            priority: Execution priority (1=highest)

        Returns:
            The created SubQuery instance
        """
        sub_query = SubQuery(query=query, rationale=rationale, priority=priority)
        self.sub_queries.append(sub_query)
        self.updated_at = datetime.now(timezone.utc)
        return sub_query

    def get_sub_query(self, sub_query_id: str) -> Optional[SubQuery]:
        """Get a sub-query by ID."""
        for sq in self.sub_queries:
            if sq.id == sub_query_id:
                return sq
        return None

    def get_source(self, source_id: str) -> Optional[ResearchSource]:
        """Get a source by ID."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def get_gap(self, gap_id: str) -> Optional[ResearchGap]:
        """Get a gap by ID."""
        for gap in self.gaps:
            if gap.id == gap_id:
                return gap
        return None

    def get_citation_map(self) -> dict[int, ResearchSource]:
        """Build a mapping from citation number to source.

        Returns:
            Dict mapping citation_number → ResearchSource for all sources
            that have an assigned citation number.
        """
        return {s.citation_number: s for s in self.sources if s.citation_number is not None}

    def source_id_to_citation(self) -> dict[str, int]:
        """Build a mapping from source ID to citation number.

        Returns:
            Dict mapping source.id → citation_number for all sources
            that have an assigned citation number.
        """
        return {s.id: s.citation_number for s in self.sources if s.citation_number is not None}

    def add_source(
        self,
        title: str,
        url: Optional[str] = None,
        source_type: SourceType = SourceType.WEB,
        snippet: Optional[str] = None,
        sub_query_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ResearchSource:
        """Add a discovered source.

        Args:
            title: Source title
            url: Source URL (optional)
            source_type: Type of source
            snippet: Brief excerpt
            sub_query_id: ID of sub-query that found this
            **kwargs: Additional fields

        Returns:
            The created ResearchSource instance
        """
        # Citation numbering uses a running counter (O(1) per add).
        # This is the SINGLE source of truth — callers must NOT assign
        # citation_number manually.
        next_citation = self.next_citation_number
        self.next_citation_number += 1
        source = ResearchSource(
            title=title,
            url=url,
            source_type=source_type,
            snippet=snippet,
            sub_query_id=sub_query_id,
            citation_number=next_citation,
            **kwargs,
        )
        self.sources.append(source)
        self.total_sources_examined += 1
        self._evict_oldest_source_content()
        self.updated_at = datetime.now(timezone.utc)
        return source

    def append_source(self, source: ResearchSource) -> ResearchSource:
        """Append a pre-constructed source, assigning it the next citation number.

        Use this when the source is already constructed (e.g., from a search
        provider) but needs a stable citation number and state tracking.

        Args:
            source: Pre-constructed ResearchSource (citation_number will be overwritten)

        Returns:
            The same source instance, with citation_number set
        """
        source.citation_number = self.next_citation_number
        self.next_citation_number += 1
        self.sources.append(source)
        self.total_sources_examined += 1
        self._evict_oldest_source_content()
        self.updated_at = datetime.now(timezone.utc)
        return source

    def _evict_oldest_source_content(self) -> None:
        """Evict content from oldest sources when the cap is exceeded.

        Keeps URL, title, citation_number, and metadata intact but clears the
        heavy ``content``, ``raw_content``, and ``snippet`` fields to bound
        memory growth.  Only fires when ``len(self.sources) > MAX_SOURCES``.
        """
        if len(self.sources) <= MAX_SOURCES:
            return
        overshoot = len(self.sources) - MAX_SOURCES
        for src in self.sources[:overshoot]:
            src.content = None
            src.raw_content = None
            src.snippet = None

    def add_supervision_message(self, message: dict[str, Any]) -> None:
        """Append a supervision message, enforcing the entry-count cap.

        When ``MAX_SUPERVISION_MESSAGES`` is exceeded, the oldest entries are
        dropped to keep serialized size bounded.
        """
        self.supervision_messages.append(message)
        if len(self.supervision_messages) > MAX_SUPERVISION_MESSAGES:
            overshoot = len(self.supervision_messages) - MAX_SUPERVISION_MESSAGES
            self.supervision_messages = self.supervision_messages[overshoot:]

    def add_topic_research_result(self, result: "TopicResearchResult") -> None:
        """Append a topic research result, enforcing the cap.

        When ``MAX_TOPIC_RESEARCH_RESULTS`` is exceeded, the oldest results
        are dropped to bound memory growth.
        """
        self.topic_research_results.append(result)
        if len(self.topic_research_results) > MAX_TOPIC_RESEARCH_RESULTS:
            overshoot = len(self.topic_research_results) - MAX_TOPIC_RESEARCH_RESULTS
            self.topic_research_results = self.topic_research_results[overshoot:]

    def add_finding(
        self,
        content: str,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        source_ids: Optional[list[str]] = None,
        sub_query_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> ResearchFinding:
        """Add a research finding.

        Args:
            content: The finding content
            confidence: Confidence level
            source_ids: Supporting source IDs
            sub_query_id: Originating sub-query ID
            category: Theme/category

        Returns:
            The created ResearchFinding instance
        """
        finding = ResearchFinding(
            content=content,
            confidence=confidence,
            source_ids=source_ids or [],
            sub_query_id=sub_query_id,
            category=category,
        )
        self.findings.append(finding)
        self.updated_at = datetime.now(timezone.utc)
        return finding

    def add_gap(
        self,
        description: str,
        suggested_queries: Optional[list[str]] = None,
        priority: int = 1,
    ) -> ResearchGap:
        """Add an identified research gap.

        Args:
            description: What information is missing
            suggested_queries: Follow-up queries to fill the gap
            priority: Priority for follow-up (1=highest)

        Returns:
            The created ResearchGap instance
        """
        gap = ResearchGap(
            description=description,
            suggested_queries=suggested_queries or [],
            priority=priority,
        )
        self.gaps.append(gap)
        self.updated_at = datetime.now(timezone.utc)
        return gap

    # =========================================================================
    # Query Helpers
    # =========================================================================

    def pending_sub_queries(self) -> list[SubQuery]:
        """Get sub-queries that haven't been executed yet."""
        return [sq for sq in self.sub_queries if sq.status == "pending"]

    def completed_sub_queries(self) -> list[SubQuery]:
        """Get successfully completed sub-queries."""
        return [sq for sq in self.sub_queries if sq.status == "completed"]

    def failed_sub_queries(self) -> list[SubQuery]:
        """Get sub-queries that failed during execution."""
        return [sq for sq in self.sub_queries if sq.status == "failed"]

    def unresolved_gaps(self) -> list[ResearchGap]:
        """Get gaps that haven't been resolved yet."""
        return [g for g in self.gaps if not g.resolved]

    # =========================================================================
    # Cost Tracking
    # =========================================================================

    def get_model_role_costs(self) -> dict[str, dict[str, Any]]:
        """Aggregate cost tracking per model role across all phase metrics.

        Scans ``phase_metrics`` for entries that have a ``role`` key in their
        metadata (set by ``execute_llm_call`` when a role is provided) and
        produces a per-role cost summary.

        Returns:
            Dict keyed by role name, each containing::

                {
                    "provider": str | None,  # last-seen provider for this role
                    "model": str | None,  # last-seen model for this role
                    "input_tokens": int,
                    "output_tokens": int,
                    "calls": int,
                }
        """
        role_costs: dict[str, dict[str, Any]] = {}
        for pm in self.phase_metrics:
            role = pm.metadata.get("role")
            if not role:
                continue
            if role not in role_costs:
                role_costs[role] = {
                    "provider": None,
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                }
            entry = role_costs[role]
            entry["provider"] = pm.provider_id or entry["provider"]
            entry["model"] = pm.model_used or entry["model"]
            entry["input_tokens"] += pm.input_tokens
            entry["output_tokens"] += pm.output_tokens
            entry["calls"] += 1
        return role_costs

    # =========================================================================
    # Phase Management
    # =========================================================================

    # Phases that advance_phase() skips over automatically.
    # PLANNING and GATHERING are deprecated — new workflows proceed
    # BRIEF → SUPERVISION directly.  Retained in the enum only for legacy
    # saved-state resume compatibility.
    _SKIP_PHASES: ClassVar[set[DeepResearchPhase]] = {
        DeepResearchPhase.PLANNING,
        DeepResearchPhase.GATHERING,
        DeepResearchPhase.ANALYSIS,
        DeepResearchPhase.REFINEMENT,
    }

    def advance_phase(self) -> DeepResearchPhase:
        """Advance to the next research phase, skipping deprecated phases.

        Phases advance in order: CLARIFICATION -> BRIEF -> SUPERVISION -> SYNTHESIS.
        PLANNING and GATHERING are automatically skipped (deprecated, legacy-resume-only).
        Does nothing if already at SYNTHESIS. The phase order is derived
        from the DeepResearchPhase enum definition order.

        Uses a while loop to handle consecutive deprecated phases correctly
        (e.g. if multiple phases are ever added to ``_SKIP_PHASES``).

        Returns:
            The new phase after advancement
        """
        phase_order = list(DeepResearchPhase)
        current_index = phase_order.index(self.phase)
        if current_index < len(phase_order) - 1:
            current_index += 1
            self.phase = phase_order[current_index]
            # Skip any consecutive deprecated phases (e.g. GATHERING)
            while self.phase in self._SKIP_PHASES and current_index < len(phase_order) - 1:
                current_index += 1
                self.phase = phase_order[current_index]
        self.updated_at = datetime.now(timezone.utc)
        return self.phase

    def should_continue_supervision(self) -> bool:
        """Check if another supervision round should occur.

        Returns True if:
        - Current supervision_round < max_supervision_rounds AND
        - There are pending sub-queries to process

        Returns:
            True if supervision should continue, False otherwise
        """
        if self.supervision_round >= self.max_supervision_rounds:
            return False
        return len(self.pending_sub_queries()) > 0

    def mark_completed(self, report: Optional[str] = None) -> None:
        """Mark the research session as completed.

        Args:
            report: Optional final report content
        """
        self.phase = DeepResearchPhase.SYNTHESIS
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        if report:
            self.report = report

    def mark_failed(self, error: str) -> None:
        """Mark the research session as failed with an error message.

        This sets completed_at to indicate the session has ended, and stores
        the failure information in metadata for status reporting.

        Args:
            error: Description of why the research failed
        """
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["failed"] = True
        self.metadata["failure_error"] = error
        self.metadata["terminal_status"] = "failed"

    def mark_cancelled(self, *, phase_state: Optional[str] = None) -> None:
        """Mark the research session as cancelled by user request.

        Distinct from mark_failed (error) and mark_interrupted (SIGTERM).
        Sets completed_at and stores cancellation context in metadata.

        Args:
            phase_state: Optional description of phase state at cancellation time
        """
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["cancelled"] = True
        self.metadata["terminal_status"] = "cancelled"
        if phase_state:
            self.metadata["cancelled_phase_state"] = phase_state

    def mark_interrupted(self, *, reason: str = "SIGTERM") -> None:
        """Mark the research session as interrupted by process signal.

        Distinct from mark_cancelled (user-initiated) and mark_failed (error).
        Used for SIGTERM and other process-level interruptions.

        Args:
            reason: Reason for interruption (default: "SIGTERM")
        """
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata["interrupted"] = True
        self.metadata["terminal_status"] = "interrupted"
        self.metadata["interrupt_reason"] = reason
        self.metadata["interrupt_phase"] = self.phase.value
        self.metadata["interrupt_iteration"] = self.iteration

    # ==========================================================================
    # Content Fidelity Tracking Methods
    # ==========================================================================

    def record_item_fidelity(
        self,
        item_id: str,
        phase: str,
        level: FidelityLevel,
        item_type: str = "source",
        reason: str = "",
        warnings: Optional[list[str]] = None,
        original_tokens: Optional[int] = None,
        final_tokens: Optional[int] = None,
    ) -> ContentFidelityRecord:
        """Record fidelity for a content item in a specific phase.

        Creates or updates the ContentFidelityRecord for the item and
        adds the phase-specific record.

        Args:
            item_id: Unique identifier for the content item
            phase: Phase name (e.g., "analysis", "synthesis")
            level: Fidelity level applied
            item_type: Type of content ("source", "finding", "gap")
            reason: Why degradation was applied
            warnings: Any warnings generated
            original_tokens: Token count before degradation
            final_tokens: Token count after degradation

        Returns:
            The ContentFidelityRecord for the item
        """
        # Create or get existing record
        if item_id not in self.content_fidelity:
            self.content_fidelity[item_id] = ContentFidelityRecord(
                item_id=item_id,
                item_type=item_type,
            )

        record = self.content_fidelity[item_id]
        record.record_phase(
            phase=phase,
            level=level,
            reason=reason,
            warnings=warnings,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )

        # Track dropped items
        if level == FidelityLevel.DROPPED and item_id not in self.dropped_content_ids:
            self.dropped_content_ids.append(item_id)

        self.updated_at = datetime.now(timezone.utc)
        return record

    def get_item_fidelity(self, item_id: str) -> Optional[ContentFidelityRecord]:
        """Get fidelity record for a content item.

        Args:
            item_id: ID of the content item

        Returns:
            ContentFidelityRecord if exists, None otherwise
        """
        return self.content_fidelity.get(item_id)

    def get_items_at_fidelity(self, level: FidelityLevel) -> list[str]:
        """Get all item IDs currently at a specific fidelity level.

        Args:
            level: Fidelity level to filter by

        Returns:
            List of item IDs at that fidelity level
        """
        return [item_id for item_id, record in self.content_fidelity.items() if record.current_level == level]

    def get_overall_fidelity_score(self) -> float:
        """Calculate an overall fidelity score for the session.

        Returns a value between 0.0 and 1.0 representing the average
        content preservation across all tracked items.

        Returns:
            Overall fidelity score (1.0 = all full fidelity, 0.0 = all dropped)
        """
        if not self.content_fidelity:
            return 1.0

        level_scores = {
            FidelityLevel.FULL: 1.0,
            FidelityLevel.CONDENSED: 0.7,
            FidelityLevel.KEY_POINTS: 0.4,
            FidelityLevel.HEADLINE: 0.2,
            FidelityLevel.TRUNCATED: 0.3,
            FidelityLevel.DROPPED: 0.0,
        }

        total_score = sum(level_scores.get(record.current_level, 0.5) for record in self.content_fidelity.values())
        return total_score / len(self.content_fidelity)

    def has_degraded_content(self) -> bool:
        """Check if any content has been degraded from full fidelity.

        Returns:
            True if any content is below FULL fidelity
        """
        return any(record.current_level != FidelityLevel.FULL for record in self.content_fidelity.values())

    def record_chunk_fidelity(
        self,
        base_id: str,
        chunk_index: int,
        phase: str,
        level: FidelityLevel,
        item_type: str = "source",
        reason: str = "",
        warnings: Optional[list[str]] = None,
        original_tokens: Optional[int] = None,
        final_tokens: Optional[int] = None,
    ) -> ContentFidelityRecord:
        """Record fidelity for a specific chunk of a content item.

        Creates a fidelity record with a stable fragment ID in the format
        "{base_id}#fragment-{N}". This allows tracking fidelity at the
        chunk level while maintaining the parent item relationship.

        Args:
            base_id: Base item ID (e.g., "src-abc123")
            chunk_index: Zero-based index of the chunk
            phase: Phase name (e.g., "analysis", "synthesis")
            level: Fidelity level applied
            item_type: Type of content ("source", "finding", "gap")
            reason: Why degradation was applied
            warnings: Any warnings generated
            original_tokens: Token count before degradation
            final_tokens: Token count after degradation

        Returns:
            The ContentFidelityRecord for the chunk
        """
        fragment_id = make_fragment_id(base_id, chunk_index)
        return self.record_item_fidelity(
            item_id=fragment_id,
            phase=phase,
            level=level,
            item_type=item_type,
            reason=reason,
            warnings=warnings,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
        )

    def get_chunk_fidelity(self, base_id: str, chunk_index: int) -> Optional[ContentFidelityRecord]:
        """Get fidelity record for a specific chunk.

        Args:
            base_id: Base item ID (e.g., "src-abc123")
            chunk_index: Zero-based index of the chunk

        Returns:
            ContentFidelityRecord if exists, None otherwise
        """
        fragment_id = make_fragment_id(base_id, chunk_index)
        return self.get_item_fidelity(fragment_id)

    def get_all_chunks_for_item(self, base_id: str) -> dict[int, ContentFidelityRecord]:
        """Get all chunk fidelity records for a base item.

        Finds all fragment IDs that derive from the given base ID and
        returns their fidelity records indexed by chunk number.

        Args:
            base_id: Base item ID (e.g., "src-abc123")

        Returns:
            Dict mapping chunk_index to ContentFidelityRecord
        """
        chunks = {}
        prefix = f"{base_id}#fragment-"
        for item_id, record in self.content_fidelity.items():
            if item_id.startswith(prefix):
                _, fragment_index = parse_fragment_id(item_id)
                if fragment_index is not None:
                    chunks[fragment_index] = record
        return chunks

    def merge_fidelity_record(self, item_id: str, other_record: ContentFidelityRecord) -> ContentFidelityRecord:
        """Merge another fidelity record into the state.

        Implements the fidelity merge rules:
        - Latest phase overwrites same-phase entry (by timestamp)
        - Prior phases are preserved for history

        If the item doesn't exist in state, adds it directly.
        If the item exists, merges phases from the other record.

        Args:
            item_id: ID of the content item
            other_record: ContentFidelityRecord to merge

        Returns:
            The merged ContentFidelityRecord
        """
        if item_id not in self.content_fidelity:
            # New item - add directly
            self.content_fidelity[item_id] = other_record
        else:
            # Existing item - merge phases
            self.content_fidelity[item_id].merge_phases_from(other_record)

        # Track dropped items
        record = self.content_fidelity[item_id]
        if record.current_level == FidelityLevel.DROPPED and item_id not in self.dropped_content_ids:
            self.dropped_content_ids.append(item_id)

        self.updated_at = datetime.now(timezone.utc)
        return record

    def get_aggregate_chunk_fidelity(self, base_id: str) -> Optional[FidelityLevel]:
        """Get the aggregate fidelity level across all chunks of an item.

        Returns the lowest (most degraded) fidelity level among all
        chunks. This represents the "worst case" fidelity for the item.

        Args:
            base_id: Base item ID

        Returns:
            Lowest FidelityLevel among chunks, or None if no chunks exist
        """
        chunks = self.get_all_chunks_for_item(base_id)
        if not chunks:
            return None

        # Order: FULL > CONDENSED > KEY_POINTS > HEADLINE > TRUNCATED > DROPPED
        level_order = [
            FidelityLevel.FULL,
            FidelityLevel.CONDENSED,
            FidelityLevel.KEY_POINTS,
            FidelityLevel.HEADLINE,
            FidelityLevel.TRUNCATED,
            FidelityLevel.DROPPED,
        ]

        worst_level = FidelityLevel.FULL
        for record in chunks.values():
            if level_order.index(record.current_level) > level_order.index(worst_level):
                worst_level = record.current_level

        return worst_level
