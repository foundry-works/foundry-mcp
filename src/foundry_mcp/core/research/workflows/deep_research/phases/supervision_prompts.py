"""Prompt builders for the supervision phase.

Extracted from ``supervision.py`` to reduce file size and isolate pure
prompt-construction logic from orchestration.  Every function here is a
pure function: it takes state/config/data and returns a string.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ResearchProfile,
)
from foundry_mcp.core.research.models.sources import ResearchMode
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    build_sanitized_context,
    sanitize_external_content,
)

# ======================================================================
# Shared helpers
# ======================================================================


def render_supervision_conversation_history(
    state: DeepResearchState,
    messages: list[dict[str, Any]],
) -> list[str]:
    """Render supervision conversation history as prompt lines.

    Shared by ``build_delegation_user_prompt`` and
    ``build_combined_think_delegate_user_prompt`` to eliminate duplication.

    Args:
        state: Current research state (unused currently, reserved for future)
        messages: ``state.supervision_messages`` list

    Returns:
        List of prompt lines (without trailing join)
    """
    _ = state  # reserved
    parts: list[str] = []
    for msg in messages:
        msg_round = msg.get("round", "?")
        msg_type = msg.get("type", "unknown")
        msg_content = msg.get("content", "")
        # Sanitize tool_result content (derived from web-scraped
        # sources) to strip prompt-injection vectors before
        # interpolating into the supervision prompt.
        if msg.get("role") == "tool_result":
            msg_content = sanitize_external_content(msg_content)
        if msg.get("role") == "assistant" and msg_type == "think":
            parts.append(f"### [Round {msg_round}] Your Gap Analysis")
            parts.append(msg_content)
            parts.append("")
        elif msg.get("role") == "assistant" and msg_type == "delegation":
            parts.append(f"### [Round {msg_round}] Your Delegation Response")
            parts.append(msg_content)
            parts.append("")
        elif msg.get("role") == "tool_result" and msg_type == "evidence_inventory":
            directive_id = msg.get("directive_id", "unknown")
            parts.append(f"### [Round {msg_round}] Evidence Inventory (directive {directive_id})")
            parts.append(msg_content)
            parts.append("")
        elif msg.get("role") == "tool_result":
            directive_id = msg.get("directive_id", "unknown")
            parts.append(f"### [Round {msg_round}] Research Findings (directive {directive_id})")
            parts.append(msg_content)
            parts.append("")
    return parts


def classify_query_complexity(state: DeepResearchState) -> str:
    """Classify the original query's complexity for directive scaling.

    Uses heuristics based on sub-query count and research brief length
    to produce a simple/moderate/complex label that guides the supervisor
    in calibrating how many directives to generate.

    Args:
        state: Current research state (uses sub_queries, research_brief,
               original_query)

    Returns:
        One of ``"simple"``, ``"moderate"``, or ``"complex"``
    """
    sub_query_count = len(state.sub_queries)
    brief = state.research_brief or state.original_query
    brief_word_count = len(brief.split())

    # High sub-query count or long brief → complex
    if sub_query_count >= 5 or brief_word_count >= 200:
        return "complex"

    # Moderate indicators
    if sub_query_count >= 3 or brief_word_count >= 80:
        return "moderate"

    return "simple"


# ======================================================================
# Combined think+delegate prompts (single-call mode)
# ======================================================================


def build_combined_think_delegate_system_prompt() -> str:
    """Build system prompt for the combined think+delegate step."""
    return """You are a research lead. Your task has two parts:

**Part 1 — Gap Analysis:** Analyze the research coverage and identify specific information gaps. Articulate:
- What was found per sub-query
- What domains and perspectives are represented
- What specific information gaps exist
- What types of sources or angles would fill those gaps

**Part 2 — Directive Generation:** Based on your gap analysis, generate research directives targeting the identified gaps.

Your response MUST follow this exact format:

<gap_analysis>
[Your detailed gap analysis here — plain text with section headings]
</gap_analysis>

```json
{
    "research_complete": false,
    "directives": [
        {
            "research_topic": "Detailed paragraph-length description of what to investigate...",
            "perspective": "What angle or perspective to approach from",
            "evidence_needed": "What specific evidence, data, or sources to seek",
            "priority": 1
        }
    ],
    "rationale": "Why these directives were chosen, referencing your gap analysis"
}
```

Guidelines:
- Set "research_complete" to true ONLY when existing coverage is sufficient across all dimensions. Premature completion leaves gaps that degrade report quality, but never completing wastes budget on diminishing returns. The threshold is "sufficient for a confident, well-sourced answer."
- Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences). Researchers receive this as their sole guidance — a vague directive produces a vague, unfocused research pass that wastes a full iteration budget.
- "priority": 1=critical gap, 2=important, 3=nice-to-have
- Maximum 5 directives per round
- Do NOT duplicate research already covered
- Your directives MUST directly address gaps from your analysis. Untargeted directives risk duplicating already-covered ground while leaving actual gaps unfilled.
- The gap_analysis section MUST come FIRST, before the JSON"""


def build_combined_think_delegate_user_prompt(
    state: DeepResearchState,
    coverage_data: list[dict[str, Any]],
) -> str:
    """Build user prompt for the combined think+delegate step."""
    ctx = build_sanitized_context(state)
    parts = [
        f"# Research Query\n{ctx['original_query']}",
        "",
    ]

    if state.research_brief and state.research_brief != state.original_query:
        parts.extend(
            [
                "## Research Brief",
                ctx["research_brief"],
                "",
            ]
        )

    parts.extend(
        [
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Supervision round: {state.supervision_round + 1}/{state.max_supervision_rounds}",
            f"- Completed sub-queries: {len(state.completed_sub_queries())}",
            f"- Total sources: {len(state.sources)}",
            "",
        ]
    )

    # Prior supervisor conversation
    if state.supervision_messages:
        parts.append("## Prior Supervisor Conversation")
        parts.append(
            "Below is your conversation history from previous rounds. "
            "Reference your prior reasoning and research findings."
        )
        parts.append("")
        parts.extend(
            render_supervision_conversation_history(
                state,
                state.supervision_messages,
            )
        )
        parts.append("---")
        parts.append("")

    # Per-query coverage
    if coverage_data:
        parts.append("## Current Research Coverage")
        for entry in coverage_data:
            parts.append(f"\n### {sanitize_external_content(entry['query'])}")
            parts.append(f"**Sources:** {entry['source_count']} | **Domains:** {entry['unique_domains']}")
            if entry.get("compressed_findings_excerpt"):
                parts.append(f"**Key findings:**\n{sanitize_external_content(entry['compressed_findings_excerpt'])}")
            elif entry.get("findings_summary"):
                parts.append(f"**Summary:** {sanitize_external_content(entry['findings_summary'])}")
        parts.append("")

    # Previously executed directives
    if state.directives:
        parts.append("## Previously Executed Directives (DO NOT repeat)")
        for d in state.directives[-10:]:
            parts.append(f"- [P{d.priority}] {sanitize_external_content(d.research_topic[:120])}")
        parts.append("")

    parts.extend(
        [
            "## Instructions",
            "1. First, write your gap analysis inside <gap_analysis> tags",
            "2. Then, if coverage is sufficient, return JSON with research_complete=true",
            "3. Otherwise, generate 1-5 detailed research directives as JSON",
            "4. Your directives must directly address gaps from your analysis",
            "",
        ]
    )

    return "\n".join(parts)


# ======================================================================
# Delegation prompts (two-call mode)
# ======================================================================


def _build_delegation_core_prompt() -> str:
    """Return the shared JSON schema, acronym rule, and JSON-only instruction.

    Used by both ``build_delegation_system_prompt`` and
    ``build_first_round_delegation_system_prompt`` to eliminate ~40%
    text duplication.
    """
    return """Your response MUST be valid JSON with this exact structure:
{
    "research_complete": false,
    "directives": [
        {
            "research_topic": "Detailed paragraph-length description of what to investigate...",
            "perspective": "What angle or perspective to approach from",
            "evidence_needed": "What specific evidence, data, or sources to seek",
            "priority": 1
        }
    ],
    "rationale": "Why these directives were chosen"
}

%GUIDELINES%

- Do NOT use acronyms or abbreviations in directive text — spell out all terms so researchers search for the correct concepts. Acronyms may be ambiguous (e.g., "ML" could mean machine learning or markup language) and produce irrelevant search results.

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""


def build_delegation_system_prompt() -> str:
    """Build system prompt for the delegation step.

    Returns:
        System prompt instructing directive generation
    """
    guidelines = """Guidelines:
- Set "research_complete" to true ONLY when existing coverage is sufficient across all dimensions
- Each directive's "research_topic" MUST be a detailed paragraph (2-4 sentences) specifying:
  - The specific topic to investigate
  - The research approach (compare, investigate, validate, survey, etc.)
  - What the researcher should focus on and what to exclude
- "perspective" should specify the angle: technical, comparative, historical, regulatory, user-focused, etc.
- "evidence_needed" should name concrete evidence types: statistics, case studies, expert opinions, benchmarks, etc.
- "priority": 1=critical gap (blocks report quality), 2=important (improves comprehensiveness), 3=nice-to-have. Priority determines execution order when budget is limited — critical gaps are addressed first because they block the report from being useful; nice-to-have gaps are only pursued if budget remains.
- Do NOT duplicate research already covered — target SPECIFIC gaps
- Directives should be complementary, not overlapping — each covers a different dimension

Directive Count Scaling:
- Simple factual gaps (single missing fact or stat): 1-2 directives maximum
- Comparison gaps (need data on specific compared elements): 1 directive per element needing more research
- Complex multi-dimensional gaps (multiple interrelated areas uncovered): 3-5 directives targeting distinct dimensions
- BIAS toward fewer, more focused directives — a single well-scoped directive beats three vague ones. Each directive spawns a full researcher agent with its own search budget; fewer, focused directives concentrate budget on the actual gaps, while many vague ones spread budget thin and produce overlapping, shallow results.
- Maximum 5 directives per round regardless of complexity. Each directive consumes a researcher agent's full budget — more than 5 per round risks exceeding the session's total budget and hitting diminishing returns before the next supervision assessment."""

    preamble = "You are a research lead delegating tasks to specialized researchers. Your task is to analyze research gaps and generate detailed research directives.\n\n"
    return preamble + _build_delegation_core_prompt().replace("%GUIDELINES%", guidelines)


def build_delegation_user_prompt(
    state: DeepResearchState,
    coverage_data: list[dict[str, Any]],
    think_output: Optional[str] = None,
) -> str:
    """Build user prompt for directive generation.

    Args:
        state: Current research state
        coverage_data: Per-sub-query coverage data
        think_output: Gap analysis from think step

    Returns:
        User prompt string
    """
    ctx = build_sanitized_context(state)
    parts = [
        f"# Research Query\n{ctx['original_query']}",
        "",
    ]

    if state.research_brief and state.research_brief != state.original_query:
        parts.extend(
            [
                "## Research Brief",
                ctx["research_brief"],
                "",
            ]
        )

    # Complexity signal for directive scaling
    complexity = classify_query_complexity(state)
    complexity_guidance = {
        "simple": "This is a **simple** query — target 1-2 focused directives for remaining gaps.",
        "moderate": "This is a **moderate** complexity query — target 2-3 directives for remaining gaps.",
        "complex": "This is a **complex** multi-dimensional query — target 3-5 directives for remaining gaps.",
    }

    parts.extend(
        [
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Supervision round: {state.supervision_round + 1}/{state.max_supervision_rounds}",
            f"- Completed sub-queries: {len(state.completed_sub_queries())}",
            f"- Total sources: {len(state.sources)}",
            f"- Query complexity: **{complexity}**",
            "",
            complexity_guidance[complexity],
            "",
        ]
    )

    # Prior supervisor conversation (accumulated across rounds)
    if state.supervision_messages:
        parts.append("## Prior Supervisor Conversation")
        parts.append(
            "Below is your conversation history from previous rounds. "
            "Reference your prior reasoning and the research findings "
            "to avoid re-delegating already-covered topics."
        )
        parts.append("")
        parts.extend(
            render_supervision_conversation_history(
                state,
                state.supervision_messages,
            )
        )
        parts.append("---")
        parts.append("")

    # Per-query coverage with compressed findings
    if coverage_data:
        parts.append("## Current Research Coverage")
        for entry in coverage_data:
            parts.append(f"\n### {sanitize_external_content(entry['query'])}")
            parts.append(f"**Sources:** {entry['source_count']} | **Domains:** {entry['unique_domains']}")
            if entry.get("compressed_findings_excerpt"):
                parts.append(f"**Key findings:**\n{sanitize_external_content(entry['compressed_findings_excerpt'])}")
            elif entry.get("findings_summary"):
                parts.append(f"**Summary:** {sanitize_external_content(entry['findings_summary'])}")
        parts.append("")

    # Gap analysis reference — the full think output is already in the
    # "Prior Supervisor Conversation" section above (Phase 6: think flows
    # through message history).  We include a lightweight instruction
    # rather than duplicating the full analysis text.
    if think_output:
        if state.supervision_messages:
            parts.extend(
                [
                    "## Gap Analysis",
                    "",
                    "Your gap analysis from this round is in the conversation "
                    "history above. Generate research directives that DIRECTLY "
                    "address the gaps you identified.",
                    "Each directive should target a specific gap with a detailed research plan.",
                    "",
                ]
            )
        else:
            # Fallback: no conversation history (shouldn't happen, but safe)
            parts.extend(
                [
                    "## Gap Analysis",
                    "",
                    "<gap_analysis>",
                    sanitize_external_content(think_output.strip()),
                    "</gap_analysis>",
                    "",
                    "Generate research directives that DIRECTLY address the gaps identified above.",
                    "Each directive should target a specific gap with a detailed research plan.",
                    "",
                ]
            )

    # Previously executed directives (to avoid repetition)
    if state.directives:
        parts.append("## Previously Executed Directives (DO NOT repeat)")
        for d in state.directives[-10:]:  # Last 10 for context
            parts.append(f"- [P{d.priority}] {sanitize_external_content(d.research_topic[:120])}")
        parts.append("")

    parts.extend(
        [
            "## Instructions",
            "1. Analyze the current coverage and gap analysis",
            "2. If all research dimensions are well-covered, set research_complete=true",
            "3. Otherwise, generate 1-5 detailed research directives targeting specific gaps",
            "4. Each directive should be a paragraph-length research assignment",
            "5. Prioritize critical gaps (priority 1) over nice-to-have improvements (priority 3)",
            "",
            "Return your response as JSON.",
        ]
    )

    return "\n".join(parts)


# ======================================================================
# First-round decomposition prompts
# ======================================================================


def build_first_round_think_system_prompt() -> str:
    """Build system prompt for the first-round decomposition think step.

    On the first supervision round (round 0) when supervisor-owned
    decomposition is active, the think step produces a decomposition
    strategy rather than gap analysis.

    Returns:
        System prompt instructing decomposition strategy generation
    """
    return (
        "You are a research strategist. Your task is to analyze a research "
        "brief and determine the best decomposition strategy for parallel "
        "research. You decide how many parallel researchers to launch, what "
        "angles they should cover, and what priorities to assign.\n\n"
        "Be strategic: simple factual queries may need only 1-2 researchers. "
        "Comparative analyses need one researcher per element being compared. "
        "Complex multi-dimensional topics need 3-5 researchers covering "
        "different facets.\n\n"
        "Before finalizing your strategy, verify:\n"
        "- No two researchers would cover substantially the same ground\n"
        "- No critical perspective is missing for the query type\n"
        "- Each researcher has a specific, actionable focus\n\n"
        "Respond in plain text with clear section headings."
    )


def build_first_round_think_prompt(state: DeepResearchState) -> str:
    """Build think prompt for first-round decomposition strategy.

    Presents the research brief and asks for a decomposition plan
    before generating directives.

    Args:
        state: Current research state with research_brief

    Returns:
        Think prompt string for decomposition strategy
    """
    ctx = build_sanitized_context(state)
    parts = [
        "# Research Decomposition Strategy\n",
        f"**Research Query:** {ctx['original_query']}\n",
    ]

    if state.research_brief and state.research_brief != state.original_query:
        parts.append(f"**Research Brief:**\n{ctx['research_brief']}\n")

    if state.clarification_constraints:
        parts.append("**Clarification Constraints:**")
        for key, value in state.clarification_constraints.items():
            parts.append(f"- {sanitize_external_content(str(key))}: {sanitize_external_content(str(value))}")
        parts.append("")

    if state.system_prompt:
        parts.append(f"**Additional Context:** {ctx['system_prompt']}\n")

    parts.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n")

    parts.extend(
        [
            "## Instructions\n",
            "You are given a research brief. Determine how to decompose this into parallel research tasks.\n",
            "Analyze the query and decide:",
            "1. **Query type**: Is this a simple factual query, a comparison, "
            "a list/ranking, or a complex multi-dimensional topic?",
            "2. **Decomposition strategy**: How many parallel researchers are needed and what angle should each cover?",
            "3. **Priorities**: Which research angles are critical (must-have) "
            "vs. important (improves comprehensiveness) vs. nice-to-have?",
            "4. **Self-critique**: Verify no redundant directives and no missing perspectives for this query type.\n",
            "Guidelines for researcher count:",
            "- Simple factual queries: 1-2 researchers",
            "- Comparisons: one researcher per comparison element",
            "- Lists/rankings: single researcher if straightforward, or one per category if complex",
            "- Complex multi-dimensional topics: 3-5 researchers covering different facets\n",
            "Output your decomposition strategy as structured analysis with clear headings.",
        ]
    )

    return "\n".join(parts)


def build_first_round_delegation_system_prompt(
    profile: Optional[ResearchProfile] = None,
) -> str:
    """Build system prompt for first-round decomposition delegation.

    Combines the standard delegation format with planning-quality
    decomposition guidance. This replaces the PLANNING phase's
    decomposition rules.

    When an academic profile is active, appends guidelines for targeting
    foundational/seminal works, recent empirical studies, per-discipline
    directives, and literature-review-section mapping.

    Args:
        profile: Active research profile (used to inject academic guidelines)

    Returns:
        System prompt instructing initial query decomposition via directives
    """
    guidelines = """Decomposition Guidelines:
- Generate 2-5 directives (aim for 3-4 typically for most queries)
- Bias toward FEWER researchers for simple queries (1-2 directives for straightforward factual questions)
- For COMPARISONS: create one directive per comparison element (e.g., "Product A vs Product B" → one directive for each product)
- For LISTS/RANKINGS: single directive if straightforward; one per category if the list spans diverse domains
- For COMPLEX multi-dimensional topics: 3-5 directives covering different facets (technical, economic, regulatory, user impact, etc.)
- Directives should be SPECIFIC enough to yield targeted search results
- Directives must cover DISTINCT aspects — no two should investigate substantially the same ground"""

    # PLAN-1 Item 5b: Academic decomposition guidelines
    if profile is not None and profile.source_quality_mode == ResearchMode.ACADEMIC:
        guidelines += _build_academic_decomposition_guidelines(profile)

    preamble = "You are a research lead performing initial query decomposition. Your task is to break down a research query into focused, parallel research directives — each assigned to a specialized researcher.\n\n"
    return preamble + _build_delegation_core_prompt().replace("%GUIDELINES%", guidelines)


def _build_academic_decomposition_guidelines(profile: ResearchProfile) -> str:
    """Build academic-specific decomposition guidelines.

    Appended to the first-round delegation system prompt when the active
    profile has ``source_quality_mode == ACADEMIC``.  Guides the supervisor
    to produce directives that map to literature review sections and target
    the evidence types academic research demands.

    Args:
        profile: Active academic research profile

    Returns:
        Guidelines text block to append to the base guidelines
    """
    parts: list[str] = [
        "\n\n**Academic Research Decomposition:**",
        "This is an academic research query. Apply these additional guidelines:",
        "",
        "- **Foundational/seminal works directive**: Include at least one directive "
        "targeting foundational and seminal works in the field. These are highly-cited "
        "papers that established key theories, frameworks, or findings. Sort results "
        "by citation count when possible.",
        "- **Recent empirical studies directive**: Include at least one directive "
        "targeting recent empirical studies (last 3-5 years) to capture the current "
        "state of knowledge. Focus on peer-reviewed journal articles with clear "
        "methodology sections.",
    ]

    # Per-discipline directives for cross-disciplinary topics
    if profile.disciplinary_scope and len(profile.disciplinary_scope) > 1:
        disciplines = ", ".join(profile.disciplinary_scope)
        parts.append(
            f"- **Cross-disciplinary coverage**: The research spans multiple disciplines "
            f"({disciplines}). Consider dedicating separate directives to each major "
            f"disciplinary perspective to ensure balanced coverage."
        )

    parts.extend([
        "- **Literature review section mapping**: Structure directives so they "
        "naturally map to literature review sections — theoretical foundations, "
        "thematic analysis, methodological approaches, and key debates/contradictions.",
        "- **Evidence types**: Each directive's \"evidence_needed\" should specify "
        "academic evidence types: peer-reviewed articles, sample sizes, effect sizes, "
        "confidence intervals, theoretical frameworks, replication status, and "
        "methodological quality indicators.",
    ])

    return "\n".join(parts)


def build_first_round_delegation_user_prompt(
    state: DeepResearchState,
    think_output: Optional[str] = None,
) -> str:
    """Build user prompt for first-round decomposition delegation.

    Args:
        state: Current research state with research_brief
        think_output: Decomposition strategy from think step

    Returns:
        User prompt string
    """
    ctx = build_sanitized_context(state)
    parts = [
        f"# Research Query\n{ctx['original_query']}",
        "",
    ]

    if state.research_brief and state.research_brief != state.original_query:
        parts.extend(
            [
                "## Research Brief",
                ctx["research_brief"],
                "",
            ]
        )

    if state.clarification_constraints:
        parts.append("## Clarification Constraints")
        for key, value in state.clarification_constraints.items():
            parts.append(f"- {sanitize_external_content(str(key))}: {sanitize_external_content(str(value))}")
        parts.append("")

    if state.system_prompt:
        parts.extend(
            [
                "## Additional Context",
                ctx["system_prompt"],
                "",
            ]
        )

    # Decomposition strategy from think step
    if think_output:
        parts.extend(
            [
                "## Decomposition Strategy",
                "",
                "<decomposition_strategy>",
                sanitize_external_content(think_output.strip()),
                "</decomposition_strategy>",
                "",
                "Generate research directives that implement the decomposition "
                "strategy above. Each directive should be a detailed, self-contained "
                "research assignment for a specialized researcher — sub-agents cannot "
                "see other agents' work, so every directive must include full context.",
                "",
            ]
        )

    parts.extend(
        [
            "## Instructions",
            "1. Decompose the research query into 2-5 focused research directives",
            "2. Each directive should target a distinct aspect of the query",
            "3. Each directive should be specific enough to yield targeted results",
            "4. Prioritize: 1=critical to the core question, 2=important for comprehensiveness, 3=supplementary",
            "",
            "Return your response as JSON.",
        ]
    )

    return "\n".join(parts)


# ======================================================================
# Critique / Revision prompts
# ======================================================================


def build_critique_system_prompt() -> str:
    """Build system prompt for the critique call (call 2 of 3).

    Instructs the LLM to evaluate a set of research directives against
    four quality criteria without revising them.
    """
    return """You are a research quality reviewer. You will receive a set of research directives generated for a query. Your task is to critique them — identify issues but do NOT revise the directives yourself.

Evaluate the directives against these four criteria:

1. **Redundancy**: Are any directives investigating the same topic from the same angle? If so, identify which ones overlap and should be merged.
2. **Coverage**: Is there a major dimension of the query that no directive addresses? If so, identify what perspective or facet is missing.
3. **Proportionality**: Given the complexity of the query, is the number of directives appropriate? A simple factual question needs 1-2 directives, not 4-5. A complex multi-dimensional topic warrants 3-5.
4. **Specificity**: Are all directives specific enough to yield targeted search results? Identify any that are too broad or vague.

For each criterion, state either:
- "PASS" — no issues found
- "ISSUE: <description>" — describe the specific problem

End your response with a summary line:
- "VERDICT: NO_ISSUES" — if all four criteria pass
- "VERDICT: REVISION_NEEDED" — if any criterion has issues

Be concise and specific. Focus on actionable feedback."""


def build_critique_user_prompt(
    state: DeepResearchState,
    directives_json: str,
) -> str:
    """Build user prompt for the critique call.

    Args:
        state: Current research state (for the original query)
        directives_json: JSON string of the initial directives
    """
    ctx = build_sanitized_context(state)
    return "\n".join(
        [
            f"# Original Research Query\n{ctx['original_query']}",
            "",
            "# Directives to Critique",
            directives_json,
            "",
            "Evaluate the directives above against the four criteria "
            "(redundancy, coverage, proportionality, specificity).",
        ]
    )


def build_revision_system_prompt() -> str:
    """Build system prompt for the revision call (call 3 of 3).

    Instructs the LLM to revise directives based on critique feedback.
    """
    return """You are a research lead revising a set of research directives based on critique feedback. Apply the critique to produce an improved directive set.

Your response MUST be valid JSON with this exact structure:
{
    "research_complete": false,
    "directives": [
        {
            "research_topic": "Detailed paragraph-length description of what to investigate...",
            "perspective": "What angle or perspective to approach from",
            "evidence_needed": "What specific evidence, data, or sources to seek",
            "priority": 1
        }
    ],
    "rationale": "How the critique was applied to improve the directives"
}

Revision Guidelines:
- MERGE directives flagged as redundant into single stronger directives
- ADD a directive for any missing coverage identified in the critique
- REMOVE excess directives if proportionality issues were flagged
- SHARPEN any directives flagged as too broad or vague
- Keep directives that were not flagged unchanged
- Maintain 2-5 directives total
- Each "research_topic" should be a detailed paragraph (2-4 sentences)

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""


def build_revision_user_prompt(
    state: DeepResearchState,
    directives_json: str,
    critique_text: str,
) -> str:
    """Build user prompt for the revision call.

    Args:
        state: Current research state (for the original query)
        directives_json: JSON string of the initial directives
        critique_text: Critique feedback from call 2
    """
    ctx = build_sanitized_context(state)
    return "\n".join(
        [
            f"# Original Research Query\n{ctx['original_query']}",
            "",
            "# Current Directives",
            sanitize_external_content(directives_json),
            "",
            "# Critique Feedback",
            sanitize_external_content(critique_text),
            "",
            "Revise the directives based on the critique above. Return the improved directive set as JSON.",
        ]
    )


# ======================================================================
# Think-step prompts
# ======================================================================


def build_think_prompt(
    state: DeepResearchState,
    coverage_data: list[dict[str, Any]],
    coverage_delta: Optional[str] = None,
) -> str:
    """Build a gap-analysis-only prompt for the think step.

    This is the think-tool equivalent: it forces the LLM to explicitly
    reason through findings before acting. The prompt asks for structured
    gap analysis WITHOUT producing follow-up queries — that happens in the
    separate act step (coverage assessment).

    The think step articulates:
    - What was found per sub-query
    - What domains/perspectives are represented
    - What perspectives or information gaps exist
    - What specific types of information would fill those gaps

    When a ``coverage_delta`` is provided (rounds > 0), it is injected
    before the per-query coverage section so the LLM can focus its analysis
    on what actually changed since the last round.

    Args:
        state: Current research state
        coverage_data: Per-sub-query coverage from _build_per_query_coverage
        coverage_delta: Optional delta summary from ``_compute_coverage_delta``

    Returns:
        Think prompt string for gap analysis
    """
    ctx = build_sanitized_context(state)
    parts = [
        "# Research Gap Analysis\n",
        f"**Original Query:** {ctx['original_query']}\n",
    ]

    if state.research_brief:
        brief_excerpt = ctx["research_brief"][:500]
        parts.append(f"**Research Brief:** {brief_excerpt}\n")

    parts.append(f"**Iteration:** {state.iteration}/{state.max_iterations}")
    parts.append(f"**Supervision Round:** {state.supervision_round + 1}/{state.max_supervision_rounds}")
    parts.append(f"**Total Sources:** {len(state.sources)}\n")

    # Coverage delta (injected before full coverage for focus)
    if coverage_delta:
        parts.append("## What Changed Since Last Round\n")
        parts.append(coverage_delta)
        parts.append("")
        parts.append(
            "*Focus your analysis on the changes above. Queries marked "
            "STILL INSUFFICIENT or NEW deserve the most attention.*"
        )
        parts.append("")

    # Per-sub-query findings and coverage
    if coverage_data:
        parts.append("## Per-Sub-Query Coverage\n")
        for entry in coverage_data:
            parts.append(f"### {sanitize_external_content(entry['query'])}")
            parts.append(f"- **Status:** {entry['status']}")
            parts.append(f"- **Sources found:** {entry['source_count']}")
            qd = entry["quality_distribution"]
            parts.append(f"- **Quality:** HIGH={qd['HIGH']}, MEDIUM={qd['MEDIUM']}, LOW={qd['LOW']}")
            parts.append(f"- **Domains:** {', '.join(entry['domain_list']) if entry['domain_list'] else 'none'}")
            if entry.get("findings_summary"):
                parts.append(f"- **Findings:** {sanitize_external_content(entry['findings_summary'])}")
            parts.append("")

    parts.extend(
        [
            "## Instructions\n",
            "Analyze the research coverage above. For EACH sub-query, articulate:",
            "1. What key information was found",
            "2. What domains and perspectives are represented",
            "3. What specific information gaps exist",
            "4. What types of sources or angles would fill those gaps\n",
            "Then provide an overall assessment of:",
            "- Which research dimensions are well-covered",
            "- Which dimensions are missing or underrepresented",
            "- What specific knowledge gaps, if addressed, would most improve the research\n",
            "DO NOT generate follow-up queries. Focus ONLY on analysis of what exists and what's missing.",
            "Be specific: name exact topics, perspectives, or data types that are absent.\n",
            "Respond in plain text with clear section headings.",
        ]
    )

    return "\n".join(parts)


def build_think_system_prompt() -> str:
    """Build system prompt for the think step LLM call.

    Returns:
        System prompt instructing analytical gap assessment
    """
    return (
        "You are a research gap analyst. Your task is to evaluate the "
        "coverage quality of completed research and identify specific "
        "information gaps. You do NOT generate follow-up queries — you "
        "only analyze what has been found and what is missing.\n\n"
        "Be specific and concise. Name exact topics, perspectives, data "
        "types, or source categories that are absent. Your analysis will "
        "be used by a separate process to generate targeted follow-up queries."
    )


# Legacy supervision prompts (query-generation model) have been removed.
# See git history for the original ``supervision_legacy.py`` if needed.
