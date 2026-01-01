# Deep Research Workflow: Multi-Agent Architecture

This document describes the multi-agent supervisor orchestration pattern used in the Deep Research workflow, including prompt templates for each specialist agent, input/output schemas for handoffs, and the state machine governing phase transitions.

## Overview

The Deep Research workflow uses a **supervisor-specialist** pattern where:

- A **Supervisor** agent orchestrates the overall research process
- **Specialist** agents handle specific phases of research
- **Think-tool pauses** occur between phases for quality evaluation
- The workflow supports **iterative refinement** based on identified gaps

### Agent Roles

| Agent | Responsibility |
|-------|---------------|
| SUPERVISOR | Orchestrates phase transitions, evaluates quality gates, decides iteration vs completion |
| PLANNER | Decomposes query into sub-queries, generates research brief, identifies key themes |
| GATHERER | Executes parallel search, handles rate limiting, deduplicates sources, validates quality |
| ANALYZER | Extracts findings from sources, assesses evidence quality, identifies contradictions |
| SYNTHESIZER | Generates coherent report sections, ensures logical flow, integrates findings |
| REFINER | Identifies knowledge gaps, generates follow-up queries, prioritizes gaps |

---

## State Machine Diagram

```
                                    ┌─────────────────────────────────────────────────────────────────┐
                                    │                    DEEP RESEARCH WORKFLOW                       │
                                    └─────────────────────────────────────────────────────────────────┘

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    PLANNING     │    │    GATHERING    │    │    ANALYSIS     │    │    SYNTHESIS    │    │   REFINEMENT    │
│   ───────────   │    │   ───────────   │    │   ───────────   │    │   ───────────   │    │   ───────────   │
│                 │    │                 │    │                 │    │                 │    │                 │
│  Agent: PLANNER │───▶│ Agent: GATHERER │───▶│ Agent: ANALYZER │───▶│Agent: SYNTHESIZR│───▶│  Agent: REFINER │
│                 │    │                 │    │                 │    │                 │    │                 │
│  Decompose      │    │  Execute search │    │  Extract        │    │  Generate       │    │  Identify gaps  │
│  query into     │    │  queries in     │    │  findings from  │    │  comprehensive  │    │  and follow-up  │
│  sub-queries    │    │  parallel       │    │  sources        │    │  report         │    │  queries        │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │                      │                      │
         ▼                      ▼                      ▼                      ▼                      ▼
    ┌─────────┐            ┌─────────┐            ┌─────────┐            ┌─────────┐            ┌─────────┐
    │  THINK  │            │  THINK  │            │  THINK  │            │  THINK  │            │  THINK  │
    │  PAUSE  │            │  PAUSE  │            │  PAUSE  │            │  PAUSE  │            │  PAUSE  │
    │─────────│            │─────────│            │─────────│            │─────────│            │─────────│
    │Evaluate │            │Evaluate │            │Evaluate │            │Evaluate │            │Evaluate │
    │planning │            │source   │            │findings │            │report   │            │gaps and │
    │quality  │            │quality  │            │quality  │            │quality  │            │iterate? │
    └────┬────┘            └────┬────┘            └────┬────┘            └────┬────┘            └────┬────┘
         │                      │                      │                      │                      │
         │ proceed              │ proceed              │ proceed              │                      │
         └──────────────────────┴──────────────────────┴──────────────────────┘                      │
                                                                              │                      │
                                                                              ▼                      │
                                                                     ┌───────────────┐               │
                                                                     │   SUPERVISOR  │               │
                                                                     │   DECISION    │               │
                                                                     │───────────────│               │
                                                                     │ iterate OR    │               │
                                                                     │ complete?     │               │
                                                                     └───────┬───────┘               │
                                                                             │                       │
                                            ┌────────────────────────────────┼───────────────────────┘
                                            │                                │
                                            ▼                                ▼
                                    ┌───────────────┐                ┌───────────────┐
                                    │   COMPLETED   │                │   ITERATE     │
                                    │───────────────│                │───────────────│
                                    │ Return final  │                │ New iteration │
                                    │ report        │                │ Start at      │
                                    └───────────────┘                │ PLANNING      │
                                                                     └───────┬───────┘
                                                                             │
                                                                             │ (increment iteration)
                                                                             │
                                                                             └───────────────────────────┐
                                                                                                         │
    ┌────────────────────────────────────────────────────────────────────────────────────────────────────┘
    │
    └──▶ Back to GATHERING (with gaps as additional context)


    Legend:
    ───────
    ─────▶  Phase transition (sequential)
    ──┬──   Decision point
       │
    THINK   Supervisor reflection/evaluation pause
    PAUSE   (hooks.think_pause callback)
```

### Phase Transition Rules

1. **PLANNING → GATHERING**: Always proceeds after think pause evaluation
2. **GATHERING → ANALYSIS**: Always proceeds after source collection
3. **ANALYSIS → SYNTHESIS**: Always proceeds after finding extraction
4. **SYNTHESIS → Decision**: Supervisor evaluates if refinement needed
5. **Decision → REFINEMENT**: If gaps exist AND iterations remaining
6. **Decision → COMPLETED**: If no gaps OR max iterations reached
7. **REFINEMENT → GATHERING**: New iteration with gap context

### Iteration Control

```
iteration = 1
while iteration <= max_iterations:
    if phase == SYNTHESIS:
        decision = supervisor.decide_iteration(state)
        if decision.should_iterate and state.has_unresolved_gaps():
            state.start_new_iteration()  # iteration++
            state.phase = GATHERING  # Reset to gathering with gaps
        else:
            break  # Complete
```

---

## Prompt Templates

### SUPERVISOR Prompt Template

The Supervisor evaluates phase quality and makes iteration decisions. It does not execute research directly but orchestrates the specialist agents.

```
SYSTEM:
You are a research supervisor responsible for orchestrating a multi-phase deep research workflow.
Your role is to:
1. Evaluate the quality of each completed phase
2. Decide whether to proceed, retry, or request additional work
3. Determine when research is complete vs needs iteration

You receive phase completion reports and must assess:
- Quality metrics (source count, finding count, confidence levels)
- Coverage gaps (are key aspects of the query addressed?)
- Iteration budget (current iteration vs maximum allowed)

Respond with a JSON decision:
{
    "action": "proceed|retry|iterate|complete",
    "rationale": "Explanation of decision",
    "quality_assessment": {
        "score": 1-10,
        "strengths": ["..."],
        "weaknesses": ["..."]
    },
    "guidance": "Instructions for next phase/iteration if applicable"
}
```

**Think Pause Prompts** (by phase):

| Phase | Reflection Prompt |
|-------|------------------|
| PLANNING | "Planning complete. Generated {n} sub-queries. Research brief: {bool}. Evaluate: Are sub-queries comprehensive? Any gaps in coverage?" |
| GATHERING | "Gathering complete. Collected {n} sources. Evaluate: Is source diversity sufficient? Quality distribution?" |
| ANALYSIS | "Analysis complete. Extracted {n} findings, identified {m} gaps. Evaluate: Are findings well-supported? Critical gaps?" |
| SYNTHESIS | "Synthesis complete. Report: {n} chars. Iteration {i}/{max}. Evaluate: Report quality? Need refinement?" |
| REFINEMENT | "Refinement complete. Gaps addressed: {n}/{total}. Evaluate: Continue iterating or finalize?" |

---

### PLANNER Prompt Template

```
SYSTEM:
You are a research planning assistant. Your task is to analyze a research query and decompose it into focused sub-queries that can be researched independently.

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

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text.

USER:
Research Query: {original_query}

Please decompose this research query into {max_sub_queries} or fewer focused sub-queries.

Consider:
1. What are the key aspects that need investigation?
2. What background information would help understand this topic?
3. What specific questions would lead to comprehensive coverage?
4. What different perspectives or sources might be valuable?

Generate the research plan as JSON.

Additional context: {system_prompt if provided}
```

---

### GATHERER Prompt Template

The Gatherer executes search operations programmatically and does not use LLM prompts directly. However, it follows these operational guidelines:

```
GATHERER OPERATIONAL GUIDELINES:

1. SEARCH EXECUTION
   - Execute sub-queries in parallel with concurrency limit
   - Use semaphore for rate limiting (max_concurrent)
   - Track seen URLs for deduplication

2. SOURCE COLLECTION
   - Collect up to max_sources_per_query per sub-query
   - Skip duplicate URLs (dedup by URL)
   - Assign source IDs and link to sub-query

3. ERROR HANDLING
   - Mark failed sub-queries with error message
   - Continue with remaining queries on individual failures
   - Log rate limit errors for retry guidance

4. QUALITY SIGNALS
   - Track source metadata (title, URL, snippet, content)
   - Note content extraction success/failure
   - Preserve source provenance for analysis phase
```

---

### ANALYZER Prompt Template

```
SYSTEM:
You are a research analyst. Your task is to analyze research sources and extract key findings, assess their quality, and identify knowledge gaps.

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

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text.

USER:
Original Research Query: {original_query}

Research Brief:
{research_brief}

Sources to Analyze:

Source 1 (ID: src-xxx):
  Title: {title}
  URL: {url}
  Snippet: {snippet}
  Content: {content truncated to 1000 chars}

[... up to 20 sources ...]

Please analyze these sources and:
1. Extract 2-5 key findings relevant to the research query
2. Assess confidence levels based on source agreement and authority
3. Identify any knowledge gaps or unanswered questions
4. Assess the quality of each source

Return your analysis as JSON.
```

---

### SYNTHESIZER Prompt Template

```
SYSTEM:
You are a research synthesizer. Your task is to combine analyzed findings into a comprehensive, well-structured research report.

Your response should be a well-formatted markdown report with:
- Executive summary (2-3 sentences)
- Key findings section with supporting evidence
- Analysis of conflicting information (if any)
- Knowledge gaps and limitations
- Conclusion with actionable insights

Guidelines:
- Organize findings thematically or by importance
- Cite source IDs when referencing specific information
- Distinguish between well-supported findings (high confidence) and preliminary insights (low confidence)
- Note any contradictions between sources
- Keep the report focused on the original research query

USER:
Original Research Query: {original_query}

Research Brief:
{research_brief}

Findings to Synthesize:

Finding 1 (confidence: high):
  {finding content}
  Sources: src-xxx, src-yyy

Finding 2 (confidence: medium):
  {finding content}
  Sources: src-zzz

[... all findings ...]

Knowledge Gaps Identified:
1. {gap description}
2. {gap description}

Iteration: {iteration}/{max_iterations}
Source Count: {n} sources examined
High-Quality Sources: {m} sources

Please synthesize these findings into a comprehensive research report addressing the original query.
```

---

### REFINER Prompt Template

```
SYSTEM:
You are a research refiner. Your task is to analyze knowledge gaps identified during research and generate follow-up queries to address them.

Your response MUST be valid JSON:
{
    "gap_analysis": [
        {
            "gap_id": "gap-xxx",
            "severity": "critical|moderate|minor",
            "addressable": true,
            "follow_up_queries": [
                {
                    "query": "Specific search query to address this gap",
                    "expected_contribution": "What this query should reveal"
                }
            ]
        }
    ],
    "iteration_recommendation": {
        "should_iterate": true,
        "rationale": "Why iteration is/isn't recommended",
        "priority_gaps": ["gap-xxx", "gap-yyy"]
    },
    "report_improvements": [
        "Suggested improvement to current report"
    ]
}

Guidelines:
- Assess each gap's impact on research completeness
- Generate specific, actionable follow-up queries
- Consider iteration budget (current vs max iterations)
- Prioritize gaps that would most improve the final report
- Recommend iteration only if gaps are significant and addressable

USER:
Original Research Query: {original_query}

Current Report Summary:
{report excerpt or summary}

Identified Knowledge Gaps:
Gap 1 (ID: gap-xxx, priority: 1):
  {description}
  Suggested queries: {existing suggestions}

Gap 2 (ID: gap-yyy, priority: 2):
  {description}

[... all gaps ...]

Research Status:
- Iteration: {iteration}/{max_iterations}
- Sources examined: {n}
- Findings extracted: {m}
- High-confidence findings: {k}

Please analyze these gaps and recommend whether to iterate or finalize the research.
```

---

## Input/Output Schemas for Agent Handoffs

### PLANNER Handoff

**Input Schema:**
```json
{
    "research_id": "string - unique session ID",
    "original_query": "string - the user's research question",
    "current_phase": "planning",
    "iteration": "number - current iteration (1-based)",
    "system_prompt": "string|null - optional custom context",
    "max_sub_queries": "number - maximum sub-queries to generate"
}
```

**Output Schema:**
```json
{
    "research_brief": "string - 2-3 sentence research approach summary",
    "sub_queries": [
        {
            "query": "string - specific search query",
            "rationale": "string - why this query matters",
            "priority": "number - 1 is highest"
        }
    ]
}
```

---

### GATHERER Handoff

**Input Schema:**
```json
{
    "research_id": "string",
    "original_query": "string",
    "current_phase": "gathering",
    "iteration": "number",
    "sub_queries": ["string - list of query strings to execute"],
    "source_types": ["string - e.g., 'web', 'academic', 'news'"],
    "max_sources_per_query": "number"
}
```

**Output Schema:**
```json
{
    "sources": [
        {
            "id": "string - unique source ID",
            "sub_query_id": "string - originating query",
            "title": "string",
            "url": "string|null",
            "snippet": "string - search result excerpt",
            "content": "string|null - extracted full content",
            "quality": "low|medium|high|unknown"
        }
    ],
    "stats": {
        "queries_executed": "number",
        "queries_failed": "number",
        "sources_collected": "number",
        "duplicates_skipped": "number"
    }
}
```

---

### ANALYZER Handoff

**Input Schema:**
```json
{
    "research_id": "string",
    "original_query": "string",
    "current_phase": "analysis",
    "iteration": "number",
    "source_count": "number - total sources to analyze",
    "high_quality_sources": "number - sources rated high quality"
}
```

**Output Schema:**
```json
{
    "findings": [
        {
            "id": "string - unique finding ID",
            "content": "string - the finding statement",
            "confidence": "low|medium|high|confirmed|speculation",
            "source_ids": ["string"],
            "category": "string|null"
        }
    ],
    "gaps": [
        {
            "id": "string - unique gap ID",
            "description": "string",
            "suggested_queries": ["string"],
            "priority": "number",
            "addressed": "boolean - false initially"
        }
    ],
    "quality_updates": [
        {
            "source_id": "string",
            "quality": "low|medium|high"
        }
    ]
}
```

---

### SYNTHESIZER Handoff

**Input Schema:**
```json
{
    "research_id": "string",
    "original_query": "string",
    "current_phase": "synthesis",
    "iteration": "number",
    "finding_count": "number",
    "gap_count": "number",
    "has_research_brief": "boolean"
}
```

**Output Schema:**
```json
{
    "report": "string - markdown-formatted research report",
    "report_metadata": {
        "sections": ["string - section headings"],
        "word_count": "number",
        "citations_count": "number",
        "confidence_summary": {
            "high": "number",
            "medium": "number",
            "low": "number"
        }
    }
}
```

---

### REFINER Handoff

**Input Schema:**
```json
{
    "research_id": "string",
    "original_query": "string",
    "current_phase": "refinement",
    "iteration": "number",
    "gaps": ["string - gap descriptions"],
    "remaining_iterations": "number - max_iterations - iteration",
    "has_report_draft": "boolean"
}
```

**Output Schema:**
```json
{
    "gap_analysis": [
        {
            "gap_id": "string",
            "severity": "critical|moderate|minor",
            "addressable": "boolean",
            "follow_up_queries": [
                {
                    "query": "string",
                    "expected_contribution": "string"
                }
            ]
        }
    ],
    "iteration_recommendation": {
        "should_iterate": "boolean",
        "rationale": "string",
        "priority_gaps": ["string - gap IDs"]
    }
}
```

---

## Think-Tool Pause Protocol

Think-tool pauses are supervisor reflection points inserted after each phase completes. They allow the supervisor to:

1. **Evaluate phase quality** before proceeding
2. **Adjust strategy** based on intermediate results
3. **Decide on phase retry** if quality is insufficient
4. **Record decisions** for traceability

### Pause Implementation

```python
# After each phase completion:
self.orchestrator.evaluate_phase_completion(state, phase)
prompt = self.orchestrator.get_reflection_prompt(state, phase)
guidance = self.hooks.think_pause(state, prompt)  # External hook
self.orchestrator.record_to_state(state)
```

### Pause Hook Interface

```python
def on_think_pause(state: DeepResearchState, prompt: str) -> Optional[str]:
    """
    Called at supervisor reflection points.

    Args:
        state: Current research state for context
        prompt: Reflection prompt from supervisor

    Returns:
        Optional guidance string for next phase
    """
    pass
```

### Decision Recording

All supervisor decisions are recorded in `state.metadata["agent_decisions"]`:

```json
{
    "agent": "supervisor",
    "action": "evaluate_phase",
    "rationale": "Planning produced 4 sub-queries. Sufficient for gathering.",
    "inputs": {
        "phase": "planning",
        "iteration": 1
    },
    "outputs": {
        "sub_query_count": 4,
        "has_research_brief": true,
        "quality_ok": true
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Integration with Workflow

The multi-agent architecture is implemented in `src/foundry_mcp/core/research/workflows/deep_research.py`:

- `AgentRole` enum defines the 6 specialist roles
- `PHASE_TO_AGENT` maps phases to responsible agents
- `AgentDecision` dataclass records all decisions
- `SupervisorOrchestrator` coordinates phase dispatch and evaluation
- `SupervisorHooks` allows external customization of decision logic

See the implementation for detailed code examples and error handling patterns.

---

## Agent Graph Specification

This section provides a formal specification of the agent graph, including explicit transitions, loop conditions, termination criteria, and tool-call contracts.

> **Note:** These tool-call schemas are **internal workflow contracts** and do not introduce new CLI actions or output formats. They define the interface between the supervisor and specialist agents within the workflow execution context.

### Graph Notation

```
Nodes:     [PHASE]     = workflow phase / agent execution
           (DECISION)  = supervisor decision point
           <GATE>      = quality gate / validation
           {ACTION}    = internal operation

Edges:     ───────►    = unconditional transition
           ─ ─ ─ ─►    = conditional transition
           ═══════►    = loop/iteration edge

Conditions: [cond]     = guard condition on edge
```

### Formal Agent Graph

```
                              ┌──────────────────────────────────────────────────────────────────────┐
                              │                        AGENT GRAPH                                   │
                              │         Deep Research Workflow State Machine                         │
                              └──────────────────────────────────────────────────────────────────────┘

     {INIT}
        │
        │ create_state(query, config)
        ▼
   ┌─────────┐
   │  IDLE   │ ◄─────────────────────────────────────────────────────────────────────────────────────┐
   └────┬────┘                                                                                       │
        │                                                                                            │
        │ start(query) OR resume(research_id)                                                        │
        ▼                                                                                            │
  ┌───────────┐         ┌───────────┐         ┌───────────┐         ┌───────────┐         ┌───────────┐
  │ PLANNING  │────────►│ GATHERING │────────►│ ANALYSIS  │────────►│ SYNTHESIS │────────►│REFINEMENT │
  │           │         │           │         │           │         │           │         │           │
  │  Planner  │         │  Gatherer │         │  Analyzer │         │Synthesizer│         │  Refiner  │
  └─────┬─────┘         └─────┬─────┘         └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
        │                     │                     │                     │                     │
        ▼                     ▼                     ▼                     ▼                     ▼
   <GATE:plan>           <GATE:srcs>           <GATE:find>           <GATE:rpt>            <GATE:gap>
        │                     │                     │                     │                     │
        │ [ok]                │ [ok]                │ [ok]                │                     │
        └─────────────────────┴─────────────────────┘                     │                     │
                                                                          ▼                     │
                                                                   (ITERATE?)◄─────────────────┘
                                                                       │
                                              ┌────────────────────────┼────────────────────────┐
                                              │                        │                        │
                                              ▼ [gaps>0 AND            ▼ [gaps=0 OR            ▼ [cancelled
                                                 iter<max]                iter>=max]              OR timeout]
                                         ┌─────────┐              ┌───────────┐           ┌─────────┐
                                         │ ITERATE │              │ COMPLETED │           │ ABORTED │
                                         └────┬────┘              └───────────┘           └─────────┘
                                              │                         │                       │
                                              │ iter++ ; reset_phase()  │                       │
                                              │                         ▼                       ▼
                                              │                    {save_report}           {save_state}
                                              │                         │                       │
                                              └─────────────────────────┴───────────────────────┘
                                                           │
                                                           ▼
                                                       [IDLE] (can resume if not completed)


    Legend:
    ───────
    <GATE:x>    Quality gate with validation (see Validation Rules section)
    (DECISION)  Supervisor decision point with tool-call
    [condition] Guard condition for transition
```

### Transition Table

| From | To | Condition | Trigger |
|------|-----|-----------|---------|
| IDLE | PLANNING | `action == "start"` | `execute(action="start", query=...)` |
| IDLE | (resume point) | `action == "resume"` | `resume_research(research_id)` |
| PLANNING | GATHERING | `gate_planning_ok` | Planning phase completes |
| GATHERING | ANALYSIS | `gate_gathering_ok` | All sub-queries executed |
| ANALYSIS | SYNTHESIS | `gate_analysis_ok` | Findings extracted |
| SYNTHESIS | (ITERATE?) | always | Synthesis completes |
| (ITERATE?) | REFINEMENT | `gaps > 0 AND iter < max` | Supervisor decides iterate |
| (ITERATE?) | COMPLETED | `gaps == 0 OR iter >= max` | Supervisor decides complete |
| REFINEMENT | GATHERING | always | `start_new_iteration()` |
| Any | ABORTED | `cancelled OR timeout` | Cancellation/timeout event |

### Loop Conditions

**Iteration Loop:**
```python
# Pseudocode for iteration decision
def should_iterate(state: DeepResearchState) -> bool:
    unresolved_gaps = [g for g in state.gaps if not g.addressed]
    can_iterate = state.iteration < state.max_iterations
    return len(unresolved_gaps) > 0 and can_iterate
```

**Loop Invariants:**
- `iteration` is monotonically increasing (1, 2, 3, ...)
- Each iteration resets phase to PLANNING but preserves:
  - All sources from previous iterations
  - All findings from previous iterations
  - Gap context for refined sub-queries
- Maximum iterations is bounded by `max_iterations` config (default: 3)

### Termination Criteria

The workflow terminates when ANY of these conditions is met:

| Condition | Terminal State | Report Generated |
|-----------|---------------|------------------|
| `gaps == 0` after synthesis | COMPLETED | Yes |
| `iteration >= max_iterations` | COMPLETED | Yes |
| User cancellation | ABORTED | Partial (if synthesis reached) |
| Timeout exceeded | ABORTED | Partial (if synthesis reached) |
| Unrecoverable error | ABORTED | No |

---

## Supervisor Tool-Call Schemas

The supervisor coordinates workflow execution through a series of tool calls. Each tool call has defined inputs, outputs, and side effects.

> **Internal Contract Notice:** These schemas define the internal communication protocol between workflow components. They are not exposed as external CLI commands or MCP tool actions.

### Tool: `dispatch_to_agent`

Dispatches work to a specialist agent for a specific phase.

**Input Schema:**
```json
{
    "tool": "dispatch_to_agent",
    "inputs": {
        "state": "DeepResearchState - current workflow state",
        "phase": "DeepResearchPhase - target phase enum",
        "agent": "AgentRole - resolved from PHASE_TO_AGENT mapping"
    }
}
```

**Output Schema:**
```json
{
    "decision": {
        "agent": "string - agent role value",
        "action": "string - e.g., 'execute_planning'",
        "rationale": "string - why this agent was selected",
        "inputs": {
            "research_id": "string",
            "original_query": "string",
            "current_phase": "string",
            "iteration": "number",
            "...phase_specific_inputs": "varies by phase"
        },
        "outputs": "null - populated after execution",
        "timestamp": "ISO8601 datetime"
    }
}
```

**Side Effects:**
- Records `AgentDecision` to orchestrator's decision log
- Triggers phase-specific agent execution

---

### Tool: `evaluate_phase_completion`

Supervisor evaluates whether a completed phase meets quality criteria.

**Input Schema:**
```json
{
    "tool": "evaluate_phase_completion",
    "inputs": {
        "state": "DeepResearchState - state after phase execution",
        "phase": "DeepResearchPhase - the phase that just completed"
    }
}
```

**Output Schema:**
```json
{
    "decision": {
        "agent": "supervisor",
        "action": "evaluate_phase",
        "rationale": "string - evaluation summary",
        "inputs": {
            "phase": "string - phase value",
            "iteration": "number"
        },
        "outputs": {
            "quality_ok": "boolean - meets threshold",
            "...phase_specific_metrics": "varies by phase"
        },
        "timestamp": "ISO8601 datetime"
    }
}
```

**Phase-Specific Output Fields:**

| Phase | Output Fields |
|-------|--------------|
| PLANNING | `sub_query_count`, `has_research_brief` |
| GATHERING | `source_count` |
| ANALYSIS | `finding_count`, `high_confidence_count` |
| SYNTHESIS | `has_report`, `report_length` |
| REFINEMENT | `unaddressed_gaps`, `should_iterate` |

---

### Tool: `decide_iteration`

Supervisor decides whether to iterate or complete the workflow.

**Input Schema:**
```json
{
    "tool": "decide_iteration",
    "inputs": {
        "state": "DeepResearchState - state after synthesis"
    }
}
```

**Output Schema:**
```json
{
    "decision": {
        "agent": "supervisor",
        "action": "decide_iteration",
        "rationale": "string - iteration decision explanation",
        "inputs": {
            "gap_count": "number - unaddressed gaps",
            "iteration": "number - current iteration",
            "max_iterations": "number - configured maximum"
        },
        "outputs": {
            "should_iterate": "boolean",
            "next_phase": "string - 'refinement' OR 'COMPLETED'"
        },
        "timestamp": "ISO8601 datetime"
    }
}
```

---

### Tool: `think_pause`

Triggers a reflection pause for supervisor evaluation between phases.

**Input Schema:**
```json
{
    "tool": "think_pause",
    "inputs": {
        "state": "DeepResearchState - current state",
        "prompt": "string - reflection prompt from get_reflection_prompt()"
    }
}
```

**Output Schema:**
```json
{
    "guidance": "string | null - optional guidance for next phase"
}
```

**Hook Integration:**
- If `SupervisorHooks._on_think_pause` is registered, the hook receives `(state, prompt)` and returns guidance
- If no hook registered, returns `null` (continue without external guidance)

---

## Tool Selection and Priority Rules

### Phase-to-Agent Resolution

Tool selection follows deterministic rules based on the current phase:

```python
PHASE_TO_AGENT: dict[DeepResearchPhase, AgentRole] = {
    DeepResearchPhase.PLANNING: AgentRole.PLANNER,
    DeepResearchPhase.GATHERING: AgentRole.GATHERER,
    DeepResearchPhase.ANALYSIS: AgentRole.ANALYZER,
    DeepResearchPhase.SYNTHESIS: AgentRole.SYNTHESIZER,
    DeepResearchPhase.REFINEMENT: AgentRole.REFINER,
}
```

**Priority Rules:**
1. Phase determines the specialist agent (no ambiguity)
2. Supervisor always evaluates between phases (mandatory)
3. Think pauses occur after every phase completion (configurable via hooks)

### Fallback Behavior

| Scenario | Fallback Action |
|----------|----------------|
| LLM provider unavailable | Return error, preserve state for retry |
| Search provider unavailable | Skip gathering, proceed with empty sources |
| JSON parse failure | Use fallback extraction (single query/finding) |
| Context window exceeded | Return error with truncation guidance |
| Timeout during phase | Mark phase failed, preserve partial results |

### Provider Selection Priority

For LLM operations:
1. Explicit `provider_id` parameter (if provided)
2. `state.planning_provider` (set at workflow start)
3. `config.default_provider` (global default)

For search operations:
1. Configured search providers in order: Tavily → Google → SemanticScholar
2. Skip unavailable providers (missing API key)
3. Fail if no providers available

---

## Validation Rules for Tool-Call Outputs

### Planning Phase Validation

```python
def validate_planning_output(state: DeepResearchState) -> ValidationResult:
    issues = []

    # Minimum sub-queries
    if len(state.sub_queries) < 2:
        issues.append("Insufficient sub-queries (minimum: 2)")

    # Maximum sub-queries
    if len(state.sub_queries) > state.max_sub_queries:
        issues.append(f"Too many sub-queries (max: {state.max_sub_queries})")

    # Research brief presence
    if not state.research_brief:
        issues.append("Missing research brief")

    # Sub-query quality
    for sq in state.sub_queries:
        if len(sq.query) < 10:
            issues.append(f"Sub-query too short: {sq.query[:20]}")

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        quality_score=min(10, len(state.sub_queries) * 2.5)
    )
```

### Gathering Phase Validation

```python
def validate_gathering_output(state: DeepResearchState) -> ValidationResult:
    issues = []

    # Minimum sources
    if len(state.sources) < 3:
        issues.append("Insufficient sources (minimum: 3)")

    # Source quality distribution
    high_quality = sum(1 for s in state.sources if s.quality == SourceQuality.HIGH)
    if high_quality == 0 and len(state.sources) > 0:
        issues.append("No high-quality sources found")

    # Sub-query completion rate
    completed = len(state.completed_sub_queries())
    total = len(state.sub_queries)
    if completed < total * 0.5:
        issues.append(f"Low sub-query completion rate: {completed}/{total}")

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        quality_score=min(10, len(state.sources) * 1.5)
    )
```

### Analysis Phase Validation

```python
def validate_analysis_output(state: DeepResearchState) -> ValidationResult:
    issues = []

    # Minimum findings
    if len(state.findings) < 2:
        issues.append("Insufficient findings (minimum: 2)")

    # Finding confidence distribution
    high_conf = sum(1 for f in state.findings if f.confidence == ConfidenceLevel.HIGH)
    if high_conf == 0 and len(state.findings) > 0:
        issues.append("No high-confidence findings")

    # Source coverage
    cited_sources = set()
    for f in state.findings:
        cited_sources.update(f.source_ids)
    coverage = len(cited_sources) / max(1, len(state.sources))
    if coverage < 0.3:
        issues.append(f"Low source citation coverage: {coverage:.0%}")

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        quality_score=min(10, len(state.findings) * 2 + high_conf)
    )
```

### Synthesis Phase Validation

```python
def validate_synthesis_output(state: DeepResearchState) -> ValidationResult:
    issues = []

    # Report presence
    if not state.report:
        issues.append("Missing report")
        return ValidationResult(valid=False, issues=issues, quality_score=0)

    # Minimum length
    if len(state.report) < 100:
        issues.append("Report too short (minimum: 100 chars)")

    # Section presence (basic structure check)
    if "##" not in state.report:
        issues.append("Report missing section structure")

    return ValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        quality_score=min(10, len(state.report) / 500)
    )
```

### Refinement Phase Validation

```python
def validate_refinement_output(state: DeepResearchState) -> ValidationResult:
    issues = []

    # Gap assessment
    unaddressed = len([g for g in state.gaps if not g.addressed])

    # Iteration budget check
    if unaddressed > 0 and state.iteration >= state.max_iterations:
        issues.append(f"Unaddressed gaps remain but iteration limit reached")

    return ValidationResult(
        valid=True,  # Refinement always valid (informational)
        issues=issues,
        quality_score=10 - min(10, unaddressed * 2)
    )
```

---

## Cancellation and Timeout Propagation

### Cancellation Flow

```
User Request                    Workflow State              Background Task
─────────────                   ──────────────              ───────────────
cancel(research_id) ──────────► state.metadata["cancelled"] = True
                                       │
                                       ▼
                                Check at phase boundaries ──► task.cancel()
                                       │                           │
                                       ▼                           ▼
                                raise CancelledError ◄──── asyncio.CancelledError
                                       │
                                       ▼
                                save_state(partial=True)
                                       │
                                       ▼
                                Return WorkflowResult(
                                    success=False,
                                    error="Research was cancelled",
                                    metadata={"cancelled": True}
                                )
```

### Timeout Propagation

```
Timeout Configuration           Timeout Monitoring          Timeout Action
─────────────────────           ──────────────────          ──────────────
task_timeout (overall) ─────► BackgroundTask.is_timed_out ──► mark_timeout()
                                       │                           │
timeout_per_operation ─────► asyncio.wait_for(coro, timeout)       │
         │                             │                           │
         ▼                             ▼                           ▼
    Per-phase timeout           asyncio.TimeoutError ──────► task.cancel()
                                       │                           │
                                       ▼                           ▼
                                Capture partial state       TaskStatus.TIMEOUT
                                       │
                                       ▼
                                Return with error metadata
```

### State Preservation on Abort

When cancellation or timeout occurs:

1. **Current phase state is preserved:**
   - Partial sub-queries (completed ones kept)
   - Partial sources (collected ones kept)
   - Partial findings (extracted ones kept)

2. **Metadata is updated:**
   ```python
   state.metadata["cancelled"] = True  # or "timeout" = True
   state.metadata["abort_phase"] = current_phase.value
   state.metadata["abort_iteration"] = iteration
   ```

3. **State is saved for potential resume:**
   ```python
   self.memory.save_deep_research(state)
   ```

4. **Resume behavior:**
   - Cancelled/timed-out sessions can be resumed with `resume_research()`
   - Resume continues from the interrupted phase
   - No duplicate work for completed sub-queries/sources

### Graceful Degradation

| Failure Point | Preserved State | Resume Behavior |
|---------------|-----------------|-----------------|
| During PLANNING | Original query only | Restart planning |
| During GATHERING | Sub-queries, partial sources | Continue remaining queries |
| During ANALYSIS | All sources, partial findings | Re-run analysis |
| During SYNTHESIS | All findings, partial report | Re-run synthesis |
| During REFINEMENT | Complete report, partial gaps | Re-evaluate refinement |

---

## External Provider Constraints

> **Read-Only Operations:** All external provider calls (search APIs, web fetches) are **read-only** operations. The workflow does not require write capabilities to external systems.

### Search Provider Operations

| Provider | Operations | Capabilities Required |
|----------|------------|----------------------|
| Tavily | `search(query)` | Read (API key) |
| Google | `search(query)` | Read (API key) |
| SemanticScholar | `search(query)` | Read (API key, optional) |

### LLM Provider Operations

| Operation | Direction | Side Effects |
|-----------|-----------|--------------|
| `prompt(system, user)` | Read (inference) | None |
| Token counting | Read | None |

### No Write Operations

The workflow explicitly does **NOT**:
- Modify external databases
- Create external resources
- Send notifications
- Trigger webhooks
- Store data outside local persistence

All state is maintained in:
- Local `ResearchMemory` persistence
- In-memory `DeepResearchState`
- Background task registry (ephemeral)
