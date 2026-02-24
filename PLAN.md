# PLAN: Deep Research — Per-Result Summarization, Supervisor-Owned Decomposition & Pipeline Simplification

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24
**Reference:** `~/GitHub/open_deep_research`
**Status:** Draft
**Depends on:** Phases 1-4 of prior plan (brief, inline compression, extract tool, delegation — all complete)

---

## Context

After completing four phases of architectural alignment (research brief, inline compression, iteration budgets, supervisor delegation), comparative analysis against `open_deep_research` (RACE 0.4344, #6 Deep Research Bench) reveals five remaining structural gaps, ranked by downstream quality impact:

1. **Per-result summarization** — open_deep_research summarizes every search result with a cheap model *before* the researcher sees it. Each result becomes a focused `<summary>` + `<key_excerpts>` block (~25-30% of original length). foundry-mcp feeds raw Tavily content/snippets to researchers, wasting context window on noise and degrading reflection quality.

2. **Supervisor-owned decomposition** — open_deep_research has no separate PLANNING phase. The supervisor's first iteration *is* the decomposition: it reads the brief, thinks, then issues `ConductResearch` calls. foundry-mcp has an explicit PLANNING phase that generates sub-queries before GATHERING, then SUPERVISION inherits results for queries it didn't write. Two decomposition owners is one too many.

3. **Post-gathering pipeline bloat** — open_deep_research goes: per-researcher compression → final report. foundry-mcp goes: inline compression → global COMPRESSION → ANALYSIS (digest/rank/select) → SYNTHESIS → optional REFINEMENT. Four extra stages add latency without clear evidence of quality improvement once per-result summarization and per-topic compression are in place.

4. **Structured output schemas at LLM boundaries** — open_deep_research uses Pydantic models as tool call schemas (`ConductResearch`, `ResearchComplete`, `ClarifyWithUser`). The LLM generates typed tool calls enforced by the API. foundry-mcp parses free-form JSON from LLM responses, which is brittle (regex, missing fields, malformed output).

5. **Over-engineered researcher reflection** — foundry-mcp's reflection has rigid rules ("STOP IMMEDIATELY if 3+ sources FROM 2+ DISTINCT DOMAINS AND ≥1 HIGH quality") baked into prompts. open_deep_research trusts the researcher to make this judgment via unstructured `think_tool` reflection plus a `ResearchComplete` tool call. The rigid rules cause premature stopping on complex topics and unnecessary continuation on simple ones.

Phases are ordered by dependency: Phase 1 is additive (no rearchitecture needed) and improves every downstream step. Phase 2 simplifies the flow for Phase 3. Phase 5 is a quick cleanup that can land alongside any other phase.

---

## Phase 1: Per-Result Summarization at Search Time

**Effort:** Medium | **Impact:** Very High
**Files:**
- `src/foundry_mcp/core/research/providers/tavily.py` (summarization integration)
- `src/foundry_mcp/core/research/providers/shared.py` (SourceSummarizer enhancements)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (result formatting)
- `src/foundry_mcp/config/research.py` (config fields)

### Problem

In open_deep_research, `tavily_search()` (utils.py:44-136) processes every search result through `summarize_webpage()` (utils.py:175-213) before the researcher sees it. The summarization:
- Uses a cheap/fast model (gpt-4.1-mini equivalent)
- Runs in parallel for all results via `asyncio.gather()`
- Produces structured output: `<summary>` (25-30% of original) + `<key_excerpts>` (max 5 quotes)
- Has 60-second timeout with fallback to raw content
- Deduplicates by URL before summarizing

The researcher receives clean, structured inputs:
```
--- SOURCE 1: Article Title ---
URL: https://...

SUMMARY:
<summary>Focused summary with key facts preserved</summary>

<key_excerpts>"Exact quote 1", "Exact quote 2"</key_excerpts>
```

In foundry-mcp, the `SourceSummarizer` infrastructure exists in `providers/shared.py` and is optionally wired into `tavily.py` (line 473-512) via `deep_research_fetch_time_summarization`, but:
- It's not the default path for topic research
- The output format doesn't match open_deep_research's structured `<summary>` + `<key_excerpts>` pattern
- The researcher ReAct loop receives `ResearchSource` objects with raw `content`/`snippet` fields
- The reflection step reasons over raw data, degrading gap analysis quality

This matters because cleaner inputs → better reflections → better refined queries → better sources → better compression → better supervision coverage assessment. It's the highest-leverage single change because it improves every downstream step.

### Design

Make per-result summarization the default path for topic research, matching open_deep_research's pattern:

1. **Summarization prompt**: Adapt open_deep_research's `summarize_webpage_prompt` (prompts.py:311-367) for the `SourceSummarizer`. The prompt preserves key facts, statistics, quotes, chronological ordering, and content-type-specific detail (news: 5W1H, scientific: methodology/results/conclusions, product: features/specs). Target ~25-30% of original length. Output: JSON with `summary` + `key_excerpts` fields.

2. **Parallel execution**: After Tavily search returns results and deduplication completes, summarize all results in parallel via `asyncio.gather()`. Each summarization uses the cheapest available model (summarization role in provider hierarchy). 60-second timeout per result; fallback to raw content on failure.

3. **Source enrichment**: `ResearchSource` objects get populated with structured summaries:
   - `content` ← formatted `<summary>...\n</summary>\n\n<key_excerpts>...</key_excerpts>` block
   - `raw_content` ← original content (preserved for extraction/compression)
   - `metadata["summarized"] = True` flag
   - `metadata["excerpts"]` ← list of key excerpt strings

4. **Researcher context formatting**: Update the researcher's search result rendering in topic_research.py to present summarized sources in the `--- SOURCE N: title ---` format that open_deep_research uses. This replaces the current raw-content display.

5. **Budget impact**: Summarization uses the cheapest model tier and runs in parallel, so latency impact is minimal (bounded by the slowest single summarization, typically <5s). Token cost is ~0.1x of a research-tier call per result.

### Changes

1. Update `SourceSummarizer` in `providers/shared.py`:
   - Adapt summarization prompt from open_deep_research's `summarize_webpage_prompt`
   - Structured output: `{"summary": str, "key_excerpts": list[str]}` (max 5 excerpts)
   - 60-second timeout with raw-content fallback
   - Content-type-aware guidance (news, scientific, opinion, product)
   - Target length: ~25-30% of original
2. Update `TavilySearchProvider` in `tavily.py`:
   - Make `_apply_source_summarization()` the default path when `deep_research_fetch_time_summarization=True`
   - Ensure parallel summarization via `asyncio.gather()` with per-result timeout
   - Format summarized content as `<summary>...</summary>\n\n<key_excerpts>...</key_excerpts>`
   - Preserve raw content in `raw_content` field
3. Update topic researcher result formatting in `topic_research.py`:
   - Present search results to researcher LLM in `--- SOURCE N: title ---\nURL: ...\nSUMMARY:\n...` format
   - When source is summarized, use formatted summary; otherwise fall back to snippet/content
4. Verify `deep_research_fetch_time_summarization: bool = True` default is active
5. Add config: `deep_research_summarization_timeout: int = 60` (seconds, per result)
6. Tests

### Validation

- Test: search results are summarized before researcher sees them
- Test: summarization runs in parallel (latency ≈ slowest single result, not sum)
- Test: 60-second timeout triggers fallback to raw content
- Test: researcher receives `--- SOURCE N ---` formatted output with summary + excerpts
- Test: `raw_content` preserved on source object for downstream extraction/compression
- Test: summarization disabled when config flag False (raw content passed through)
- Test: deduplication happens before summarization (no wasted summarization calls)
- Test: summarization failure is non-fatal (individual result falls back to raw)
- Test: reflection quality improves with summarized inputs (qualitative, manual inspection)

---

## Phase 2: Supervisor-Owned Decomposition (Merge Planning into Supervision)

**Effort:** Medium-High | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (first-round decomposition)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py` (deprecate/simplify)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (phase flow changes)
- `src/foundry_mcp/core/research/models/deep_research.py` (state/phase adjustments)
- `src/foundry_mcp/config/research.py` (config)

### Problem

open_deep_research has no PLANNING phase. The `supervisor` function (deep_researcher.py:178-224) receives the research brief and decomposes it on its first iteration by issuing `ConductResearch` tool calls. The supervisor owns all decomposition because it's the same entity that will later assess coverage and delegate follow-up research. This alignment matters: the decomposer and the assessor share mental models.

foundry-mcp has an explicit PLANNING phase (`planning.py`, 740 lines) that:
1. Optionally refines the query (now redundant with BRIEF phase)
2. Decomposes into 2-5 sub-queries with rationale and priority
3. Optionally self-critiques the decomposition

Then GATHERING executes those sub-queries, then SUPERVISION inherits the results. The supervisor must infer the planning phase's intent from sub-query text it didn't write. When the supervisor generates follow-up directives (Phase 4), it's layering *its* decomposition on top of *planning's* decomposition — two mental models, potential redundancy, and no single owner of research strategy.

This also prevents adaptive decomposition: the planning phase commits to 2-5 sub-queries upfront, before any research happens. open_deep_research's supervisor can start with one broad researcher, see what comes back, then decompose further — or start with five parallel researchers if the brief clearly warrants it.

### Design

Make the supervisor's first iteration the decomposition step. The PLANNING phase becomes optional (kept for backward compatibility but skipped by default):

1. **First-round decomposition**: When supervision starts with no prior research results (first round, `state.supervision_round == 0`), the supervisor's think step produces the initial decomposition. The delegate step generates `ResearchDirective` objects that serve as the initial sub-queries. This reuses the Phase 4 delegation infrastructure directly.

2. **Planning prompt absorption**: The planning phase's decomposition guidance (query specificity, coverage breadth, priority assignment) moves into the supervision system prompt's first-round section. The supervisor prompt already handles gap-driven delegation; first-round delegation is simply "gap = everything, initial coverage = nothing."

3. **Self-critique integration**: Planning's self-critique step (checking for redundancy, missing perspectives, scope issues) becomes part of the supervisor's think-before-delegate step. The think step already articulates gaps; for round 0, it articulates the decomposition strategy.

4. **Phase flow simplification**:
   - Before: BRIEF → PLANNING → GATHERING → SUPERVISION → ...
   - After: BRIEF → SUPERVISION (round 0 = decompose + gather) → SUPERVISION (round 1+ = assess + delegate) → ...
   - GATHERING phase is subsumed into supervision's execute step (already the case with delegation model)

5. **Backward compatibility**: `deep_research_supervisor_owned_decomposition: bool = True` config flag. When False, existing PLANNING → GATHERING → SUPERVISION flow preserved. Planning phase code retained but not in default path.

### Changes

1. Update `_execute_supervision_delegation_async()` in `supervision.py`:
   - Detect first round (`state.supervision_round == 0` and no prior topic results)
   - First-round think prompt: "You are given a research brief. Determine how to decompose this into parallel research tasks."
   - First-round delegate prompt: includes planning-quality decomposition guidance (specificity, breadth, priority)
   - Subsequent rounds: unchanged (gap-driven delegation)
2. Update supervision system prompt with first-round decomposition guidance:
   - Absorb planning.py's decomposition rules: "Bias toward single agent for simple queries, parallelize for comparisons, per-element agents for lists"
   - Include self-critique: "Before delegating, verify no redundant directives and no missing perspectives"
3. Update `workflow_execution.py` phase flow:
   - When `deep_research_supervisor_owned_decomposition=True`: skip PLANNING and GATHERING phases, enter SUPERVISION directly after BRIEF
   - Supervision round 0 handles initial decomposition + execution
   - When False: preserve existing PLANNING → GATHERING → SUPERVISION flow
4. Update `DeepResearchPhase` transitions:
   - BRIEF → SUPERVISION (when supervisor-owned decomposition enabled)
   - BRIEF → PLANNING (when disabled, backward compat)
5. Add config: `deep_research_supervisor_owned_decomposition: bool = True`
6. Deprecation notice on PLANNING phase (kept for backward compat, not in default path)
7. Tests

### Validation

- Test: supervisor's first round produces initial decomposition from research brief
- Test: first-round directives have the quality of planning-phase sub-queries (coverage, specificity)
- Test: supervisor adapts decomposition based on first-round results (didn't commit upfront)
- Test: full workflow runs without PLANNING/GATHERING phases (BRIEF → SUPERVISION → SYNTHESIS)
- Test: backward compat — existing flow works when config flag disabled
- Test: no regression in decomposition quality (compare sub-queries from planning vs supervisor)
- Test: supervisor round count includes decomposition round (round 0)
- Test: self-critique happens in think step (redundancy check, missing perspectives)

---

## Phase 3: Collapse Post-Gathering Pipeline

**Effort:** Medium | **Impact:** Medium-High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (phase flow)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (direct from compressed findings)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` (global pass adjustments)
- `src/foundry_mcp/core/research/models/deep_research.py` (phase enum)
- `src/foundry_mcp/config/research.py` (config)

### Problem

open_deep_research's post-research pipeline is: per-researcher compression → final report generation. Two steps.

foundry-mcp's post-research pipeline is:
1. **Global COMPRESSION** — cross-topic deduplication of already-compressed per-topic findings
2. **ANALYSIS** — digest step (rank sources, select top N, run DocumentDigestor on each)
3. **SYNTHESIS** — merge findings, generate report
4. **REFINEMENT** (optional) — improve report, potentially loop back to GATHERING

With per-result summarization (Phase 1) and inline per-topic compression (already implemented), the inputs to report generation are already clean, deduplicated, and well-cited. The global compression pass deduplicates content that's already been compressed. The analysis digest step re-processes sources that have already been summarized at search time and compressed at topic level. The refinement loop adds another full iteration for marginal gains.

Each extra stage adds: one LLM call (latency), token cost, and risk of information loss through over-compression. The question is whether the quality delta justifies 3-4x the post-research latency.

### Design

Short-circuit the default pipeline from supervision completion to synthesis, using per-topic compressed findings directly:

1. **Default path**: SUPERVISION complete → SYNTHESIS (final report). Synthesis receives per-topic `compressed_findings` and produces the final report. No intermediate COMPRESSION, ANALYSIS, or REFINEMENT phases.

2. **Synthesis input change**: Instead of consuming `state.findings` (produced by analysis), synthesis consumes per-topic `compressed_findings` directly from `state.topic_research_results`. The synthesis prompt assembles compressed findings with section headers per topic.

3. **Global compression opt-in**: Keep the global COMPRESSION phase but make it optional. Useful for very large research tasks (20+ topics) where cross-topic deduplication has value. Config flag: `deep_research_enable_global_compression: bool = False` (default off).

4. **Analysis digest opt-in**: Keep the ANALYSIS phase but make it optional. Useful when source-level ranking matters (e.g., only top 10 of 50 sources should inform the report). Config: `deep_research_enable_analysis_digest: bool = False` (default off).

5. **Refinement opt-in**: Keep REFINEMENT but default off. Config: `deep_research_enable_refinement: bool = False`.

6. **Token limit handling**: Adapt open_deep_research's progressive retry pattern for synthesis. If the final report generation exceeds token limits, retry with progressively truncated compressed findings (reduce by 10% each attempt, max 3 retries).

### Changes

1. Update `workflow_execution.py` phase flow:
   - Default path: after SUPERVISION completes → skip to SYNTHESIS
   - COMPRESSION, ANALYSIS, REFINEMENT only run when their respective config flags are True
   - Phase transition logic: check config flags to determine next phase
2. Update synthesis phase to accept compressed findings directly:
   - Build synthesis prompt from `state.topic_research_results[].compressed_findings`
   - Format: one section per topic with compressed findings and source citations
   - Include research brief as context
   - Fall back to `state.findings` when available (backward compat with analysis phase)
3. Add progressive token-limit retry to synthesis:
   - On token limit exceeded: truncate compressed findings by 10%, retry
   - Max 3 retries before graceful failure
   - Provider-specific error detection (adapt from open_deep_research's patterns)
4. Add config defaults:
   - `deep_research_enable_global_compression: bool = False`
   - `deep_research_enable_analysis_digest: bool = False`
   - `deep_research_enable_refinement: bool = False`
5. Keep all existing phase code intact (just not in default path)
6. Tests

### Validation

- Test: default pipeline is BRIEF → SUPERVISION → SYNTHESIS (no intermediate phases)
- Test: synthesis produces quality reports from compressed findings alone
- Test: token limit retry works (progressive truncation)
- Test: global compression activates when config flag True
- Test: analysis digest activates when config flag True
- Test: refinement activates when config flag True
- Test: backward compat — full pipeline works when all flags enabled
- Test: no information loss in direct-to-synthesis path (citations preserved)
- Test: latency reduction measurable (fewer LLM calls in default path)

---

## Phase 4: Structured Output Schemas at LLM Boundaries

**Effort:** Medium | **Impact:** Medium-High
**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py` (Pydantic schemas)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (delegation outputs)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (reflection outputs)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` (brief output)

### Problem

open_deep_research uses Pydantic models as structured tool call schemas at every LLM decision point:
- `ConductResearch(research_topic: str)` — delegation
- `ResearchComplete()` — completion signal
- `ClarifyWithUser(need_clarification: bool, question: str, verification: str)` — clarification
- `ResearchQuestion(research_brief: str)` — brief generation

The LLM generates typed tool calls enforced by the API provider. Parsing is handled by the framework.

foundry-mcp prompts the LLM to produce free-form JSON and then parses it with custom `_parse_*_response()` functions that use regex and manual field extraction. This is brittle:
- Missing fields cause KeyError or silent defaults
- Malformed JSON requires fallback heuristics
- No API-level enforcement of output structure
- Extra parsing code to maintain per decision point

### Design

Replace free-form JSON parsing with structured output schemas at the highest-impact LLM boundaries:

1. **Supervision delegation**: Define `DelegationResponse` schema:
   ```python
   class DelegationResponse(BaseModel):
       research_complete: bool = False
       directives: list[ResearchDirective] = []
       rationale: str
   ```
   Use `execute_structured_llm_call()` (already exists in lifecycle module) with this schema.

2. **Researcher reflection**: Define `ReflectionDecision` schema:
   ```python
   class ReflectionDecision(BaseModel):
       continue_searching: bool
       research_complete: bool = False
       refined_query: Optional[str] = None
       urls_to_extract: list[str] = []
       rationale: str
   ```

3. **Research brief**: Define `ResearchBrief` schema:
   ```python
   class ResearchBrief(BaseModel):
       research_brief: str
       scope_boundaries: Optional[str] = None
       source_preferences: Optional[str] = None
   ```

4. **Supervision think**: Keep as unstructured text (think steps benefit from free-form reasoning).

5. **Incremental adoption**: Start with supervision delegation (highest cost of parse failure), then researcher reflection, then brief. Each can be toggled independently.

### Changes

1. Add Pydantic schemas to `models/deep_research.py`:
   - `DelegationResponse` (replaces JSON parsing in `_parse_delegation_response()`)
   - `ReflectionDecision` (replaces JSON parsing in `_parse_reflection_decision()`)
   - `ResearchBriefOutput` (replaces JSON parsing in brief phase)
2. Update `_execute_supervision_delegation_async()` in supervision.py:
   - Use `execute_structured_llm_call()` with `DelegationResponse` schema
   - Remove `_parse_delegation_response()` manual parsing
   - Keep graceful fallback on schema parse failure (single directive from gap text)
3. Update `_topic_reflect()` in topic_research.py:
   - Use structured output for reflection decisions
   - Remove manual JSON parsing with regex
   - Keep fallback: on parse failure, default to `continue_searching=True`
4. Update `_execute_brief_async()` in brief.py:
   - Use structured output for brief generation
   - Remove manual parsing
5. Tests

### Validation

- Test: delegation response parsed via structured output (no regex)
- Test: reflection decision parsed via structured output
- Test: brief parsed via structured output
- Test: schema validation catches missing fields (vs silent defaults)
- Test: fallback behavior when structured output fails (graceful degradation)
- Test: no regression in output quality (same fields, more reliable parsing)
- Test: `execute_structured_llm_call()` retry logic works with new schemas

---

## Phase 5: Simplify Researcher Reflection

**Effort:** Low | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (reflection prompt + loop)

### Problem

foundry-mcp's researcher reflection has rigid rules in the system prompt:
```
STOP IMMEDIATELY if: 3+ sources FROM 2+ DISTINCT DOMAINS AND ≥1 HIGH quality source
STOP if: 3+ relevant sources from distinct domains (even without high-quality)
ADEQUATE (no continue) if: 2+ sources but same domain or single perspective
CONTINUE if: <2 relevant sources found
```

These rules are:
- **Too aggressive for complex topics**: 3 sources from 2 domains is a low bar for PhD-level research tasks. Deep Research Bench queries often need 8-10 diverse sources.
- **Too permissive for simple topics**: The rules don't account for content quality — 2 mediocre sources trigger CONTINUE even when the topic is simple.
- **Inflexible**: The rules can't adapt to context (e.g., a comparison topic needs sources for *each* side, not just 3 total).

open_deep_research uses unstructured `think_tool` reflection: "After each search, pause and assess with think_tool." The researcher decides when it's done by calling `ResearchComplete`. The prompt provides guidance ("simple queries: 2-3 searches, complex: 5 max") but doesn't mandate rigid thresholds.

### Design

Replace rigid reflection rules with adaptive, think-tool-based reflection:

1. **Remove threshold rules**: Drop the "STOP IMMEDIATELY if 3+ sources" rules from the reflection system prompt.
2. **Add research guidance**: Replace with open_deep_research-style guidance: "Start with broader searches, then narrow as gaps emerge. For simple queries, 2-3 searches are usually sufficient. For complex topics, use up to 5-7 searches. Stop when you are confident the findings address the research question."
3. **ResearchComplete signal**: The researcher calls `ResearchComplete` (from Phase 4 structured outputs) when satisfied, rather than a JSON `research_complete: true` field parsed from reflection.
4. **Think-tool emphasis**: The reflection prompt emphasizes reasoning about what was found vs. what's needed, not counting sources and domains.

### Changes

1. Update reflection system prompt in `topic_research.py`:
   - Remove rigid threshold rules (3+ sources, 2+ domains, HIGH quality)
   - Add adaptive guidance: "Assess whether findings substantively answer the research question"
   - Add scaling guidance: "Simple factual queries: 2-3 searches. Comparative analysis: 4-6 searches. Complex multi-dimensional: up to budget limit."
   - Preserve `urls_to_extract` recommendation guidance
2. Simplify reflection parsing:
   - Primary signal: `continue_searching` boolean and `refined_query`
   - `research_complete` derived from researcher calling `ResearchComplete` tool (Phase 4)
   - Remove source-count and domain-count injection into reflection context (the researcher can see its own sources)
3. Update early-exit heuristic:
   - Keep budget-based hard cap (`max_tool_calls`)
   - Remove metadata-threshold early exit (let the LLM decide)
   - Keep "no new sources found" early exit (prevents infinite loops)
4. Tests

### Validation

- Test: researcher continues searching on complex topics beyond 3-source threshold
- Test: researcher stops early on simple topics without hitting hard rules
- Test: `ResearchComplete` signal terminates research loop
- Test: budget hard cap still enforced
- Test: "no new sources" early exit still works
- Test: reflection provides richer rationale (qualitative, not just threshold checks)
- Test: no regression on simple research tasks (still completes efficiently)

---

## Dependency Graph

```
Phase 1 (Per-Result Summarization)
    ↓ improves inputs for
Phase 2 (Supervisor-Owned Decomposition)
    ↓ simplifies flow for
Phase 3 (Collapse Pipeline)

Phase 4 (Structured Outputs) — independent, can land alongside any phase
Phase 5 (Simplify Reflection) — independent, can land alongside any phase
```

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Summarization latency adds to search time | Parallel execution, 60s timeout, cheap model. Net latency ≈ 3-5s per search iteration. |
| 1 | Information loss in summarization | Preserve `raw_content` on source; downstream extraction still has full content. |
| 2 | Decomposition quality degrades without dedicated planning | Planning's decomposition rules absorbed into supervisor prompt. Self-critique integrated into think step. |
| 2 | Breaking change for existing workflows | Config flag with backward-compat path. Planning phase code retained. |
| 3 | Reports degrade without analysis/refinement passes | Per-result summarization + per-topic compression produce clean inputs. Token-limit retry handles overflow. |
| 4 | Structured output not supported by all providers | Fallback to free-form JSON parsing when provider doesn't support structured output. |
| 5 | Researcher over-searches without rigid rules | Budget hard cap (10 tool calls) prevents runaway. "No new sources" exit prevents loops. |
