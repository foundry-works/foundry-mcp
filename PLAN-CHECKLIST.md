# PLAN-CHECKLIST: Deep Research — Per-Result Summarization, Supervisor-Owned Decomposition & Pipeline Simplification

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24

---

## Phase 1: Per-Result Summarization at Search Time

- [x] **1.1** Update `SourceSummarizer` prompt in `providers/shared.py`
  - Adapt from open_deep_research's `summarize_webpage_prompt` (prompts.py:311-367)
  - Structured output: `{"summary": str, "key_excerpts": list[str]}` (max 5 excerpts)
  - Content-type-aware guidance:
    - News articles: who, what, when, where, why, how
    - Scientific: methodology, results, conclusions
    - Opinion pieces: main arguments and supporting points
    - Product pages: key features, specs, unique selling points
  - Target length: ~25-30% of original (unless already concise)
  - Preserve: key facts, statistics, data points, important quotes, dates, names, locations
  - Maintain chronological order for time-sensitive content
- [x] **1.2** Add 60-second per-result summarization timeout
  - Wrap each `SourceSummarizer` call with `asyncio.wait_for(timeout=60)`
  - On timeout: log warning, return original content unmodified
  - On any exception: log warning, return original content (non-fatal)
- [x] **1.3** Update `TavilySearchProvider._apply_source_summarization()`
  - Ensure parallel summarization via `asyncio.gather()` for all results
  - Deduplication by URL *before* summarization (no wasted calls)
  - Format output: `<summary>...</summary>\n\n<key_excerpts>...</key_excerpts>`
  - Preserve raw content in `ResearchSource.raw_content` field
  - Set `metadata["summarized"] = True` on summarized sources
  - Set `metadata["excerpts"]` with list of key excerpt strings
- [x] **1.4** Make fetch-time summarization the default for topic research
  - Verify `deep_research_fetch_time_summarization: bool = True` is active default
  - Ensure `_apply_source_summarization()` is called in the topic research search path
  - Confirm summarization uses cheapest model tier (summarization role in provider hierarchy)
- [x] **1.5** Update researcher result formatting in `topic_research.py`
  - Present search results to researcher LLM in structured format:
    ```
    --- SOURCE N: {title} ---
    URL: {url}

    SUMMARY:
    <summary>{summary}</summary>

    <key_excerpts>{excerpts}</key_excerpts>
    ```
  - When source is summarized (`metadata["summarized"]=True`): use formatted summary
  - When not summarized: fall back to `snippet` or truncated `content`
- [x] **1.6** Add config field: `deep_research_summarization_timeout: int = 60`
  - Per-result timeout in seconds
  - Add `from_toml_dict()` parsing
- [x] **1.7** Add tests for per-result summarization
  - Test: search results are summarized before researcher sees them
  - Test: summarization runs in parallel (mock confirms concurrent calls)
  - Test: 60-second timeout triggers raw-content fallback
  - Test: researcher receives `--- SOURCE N ---` formatted output
  - Test: `raw_content` preserved on source object
  - Test: summarization disabled when `deep_research_fetch_time_summarization=False`
  - Test: deduplication happens before summarization
  - Test: individual summarization failure is non-fatal (other results still summarized)
  - Test: `metadata["summarized"]` flag set correctly
  - Test: `metadata["excerpts"]` populated with key excerpts list

---

## Phase 2: Supervisor-Owned Decomposition

- [x] **2.1** Add first-round detection to supervision delegation loop
  - In `_execute_supervision_delegation_async()`: detect `state.supervision_round == 0` with no prior topic results
  - Branch to first-round-specific think + delegate prompts
  - Subsequent rounds: unchanged gap-driven delegation
- [x] **2.2** Create first-round think prompt
  - "You are given a research brief. Determine how to decompose this into parallel research tasks."
  - Include brief text, clarification constraints, date context
  - Output: decomposition strategy (how many researchers, what angles, what priorities)
  - Include self-critique: "Verify no redundant directives and no missing perspectives"
- [x] **2.3** Create first-round delegate prompt
  - Absorb planning.py decomposition rules:
    - Bias toward single agent for simple queries
    - Parallelize for comparisons (one per comparison element)
    - For lists/rankings: single agent if straightforward
    - 2-5 directives for typical queries
  - Include priority assignment guidance (1=critical, 2=important, 3=nice-to-have)
  - Include query specificity guidance: each directive should be specific enough to yield targeted results
- [x] **2.4** Update `workflow_execution.py` phase flow
  - When `deep_research_supervisor_owned_decomposition=True`:
    - After BRIEF phase → skip PLANNING and GATHERING → enter SUPERVISION directly
    - Supervision round 0 handles decomposition + initial research execution
  - When `deep_research_supervisor_owned_decomposition=False`:
    - Preserve existing flow: BRIEF → PLANNING → GATHERING → SUPERVISION
  - Update `_determine_next_phase()` logic for new transitions
- [x] **2.5** Update `DeepResearchPhase` transitions
  - Add BRIEF → SUPERVISION transition (when supervisor-owned decomposition enabled)
  - Keep BRIEF → PLANNING transition (backward compat)
  - Ensure phase advancement logic handles both paths
- [x] **2.6** Add config: `deep_research_supervisor_owned_decomposition: bool = True`
  - Default True (new behavior)
  - Add `from_toml_dict()` parsing
  - Add deprecation log when PLANNING phase runs in supervisor-owned mode (should be skipped)
- [x] **2.7** Add deprecation notice to planning phase
  - Log info message when planning phase is skipped: "Planning phase skipped — decomposition handled by supervisor (round 0)"
  - Keep planning.py code intact for backward compat
  - Do NOT delete planning phase code
- [x] **2.8** Add tests for supervisor-owned decomposition
  - Test: supervisor's first round produces initial decomposition from research brief
  - Test: first-round directives match planning-quality (coverage, specificity, priority)
  - Test: supervisor adapts decomposition after first-round results
  - Test: full workflow BRIEF → SUPERVISION → SYNTHESIS works end-to-end
  - Test: backward compat — `supervisor_owned_decomposition=False` uses PLANNING → GATHERING flow
  - Test: supervisor round 0 counted toward `max_supervision_rounds`
  - Test: self-critique integrated (redundancy detection in think step)
  - Test: simple query → fewer directives (1-2); complex query → more (3-5)

---

## Phase 3: Collapse Post-Gathering Pipeline

- [x] **3.1** Update `workflow_execution.py` default phase flow
  - Default: after SUPERVISION completes → skip to SYNTHESIS
  - COMPRESSION phase only runs when `deep_research_enable_global_compression=True`
  - ANALYSIS phase only runs when `deep_research_enable_analysis_digest=True`
  - REFINEMENT phase only runs when `deep_research_enable_refinement=True`
  - Update `_determine_next_phase()` to check config flags
- [x] **3.2** Update synthesis to consume compressed findings directly
  - In `_build_synthesis_user_prompt()`:
    - Read from `state.topic_research_results[].compressed_findings`
    - Format: one section per topic with findings and source citations
    - Include research brief as context header
    - Fall back to `state.findings` when available (backward compat with analysis)
  - In `_build_synthesis_system_prompt()`:
    - Include report structure guidance (from open_deep_research's `final_report_generation_prompt`)
    - Comparison reports: intro, per-element sections, comparison, conclusion
    - List reports: direct list or per-item sections
    - General: overview, key findings by theme, conclusion
    - Maintain inline citations `[1][2]` referencing source list
- [x] **3.3** Add progressive token-limit retry to synthesis
  - On token limit exceeded error:
    - Detect provider-specific error patterns (OpenAI, Anthropic, etc.)
    - Truncate compressed findings by 10%
    - Retry synthesis call
    - Max 3 retries before graceful failure with partial report
  - Log each retry with truncation percentage
  - NOTE: Already handled by existing `execute_llm_call` lifecycle helper (3 retries, tiered truncation strategies)
- [x] **3.4** Add config defaults
  - `deep_research_enable_global_compression: bool = False`
  - `deep_research_enable_analysis_digest: bool = False`
  - `deep_research_enable_refinement: bool = False`
  - Add `from_toml_dict()` parsing for all three
  - Keep existing phase code intact (opt-in, not deleted)
- [x] **3.5** Add tests for collapsed pipeline
  - Test: default pipeline is BRIEF → SUPERVISION → SYNTHESIS
  - Test: synthesis produces quality reports from compressed findings alone
  - Test: token-limit retry works (progressive truncation, max 3 attempts)
  - Test: global compression activates when `enable_global_compression=True`
  - Test: analysis digest activates when `enable_analysis_digest=True`
  - Test: refinement activates when `enable_refinement=True`
  - Test: full pipeline works when all three flags enabled (backward compat)
  - Test: citations preserved in direct-to-synthesis path
  - Test: synthesis handles missing `compressed_findings` gracefully (falls back)
  - Test: latency reduction measurable (fewer phases = fewer LLM calls)

---

## Phase 4: Structured Output Schemas at LLM Boundaries

- [ ] **4.1** Add Pydantic schemas to `models/deep_research.py`
  - `DelegationResponse`:
    ```python
    class DelegationResponse(BaseModel):
        research_complete: bool = False
        directives: list[ResearchDirective] = []
        rationale: str
    ```
  - `ReflectionDecision`:
    ```python
    class ReflectionDecision(BaseModel):
        continue_searching: bool
        research_complete: bool = False
        refined_query: Optional[str] = None
        urls_to_extract: list[str] = Field(default_factory=list, max_length=2)
        rationale: str
    ```
  - `ResearchBriefOutput`:
    ```python
    class ResearchBriefOutput(BaseModel):
        research_brief: str
        scope_boundaries: Optional[str] = None
        source_preferences: Optional[str] = None
    ```
- [ ] **4.2** Update supervision delegation to use structured output
  - In `_execute_supervision_delegation_async()`:
    - Replace `_parse_delegation_response()` manual JSON parsing with `execute_structured_llm_call()` using `DelegationResponse` schema
    - Remove regex-based parsing fallbacks
    - Keep graceful degradation: on structured output failure, generate single directive from gap text
  - Verify `execute_structured_llm_call()` supports the schema (check retry logic)
- [ ] **4.3** Update researcher reflection to use structured output
  - In `_topic_reflect()`:
    - Replace manual JSON parsing with structured output using `ReflectionDecision` schema
    - Remove regex + manual field extraction
    - Keep fallback: on parse failure, default to `continue_searching=True`
  - Verify structured output works with reflection-tier (cheap) model
- [ ] **4.4** Update brief generation to use structured output
  - In `_execute_brief_async()`:
    - Use `ResearchBriefOutput` schema
    - Remove manual parsing
    - Keep fallback: on failure, use original query as brief
- [ ] **4.5** Add provider compatibility handling
  - Check if current provider supports structured output / tool use
  - When not supported: fall back to free-form JSON parsing (existing code path)
  - Log info when using fallback path
- [ ] **4.6** Add tests for structured outputs
  - Test: delegation response parsed via structured output (no regex)
  - Test: reflection decision parsed via structured output
  - Test: brief parsed via structured output
  - Test: schema validation rejects malformed responses (missing required fields)
  - Test: fallback to free-form parsing when structured output fails
  - Test: fallback to free-form parsing when provider doesn't support structured output
  - Test: `execute_structured_llm_call()` retry logic works with new schemas
  - Test: no regression in output quality (same fields, same values)

---

## Phase 5: Simplify Researcher Reflection

- [ ] **5.1** Update reflection system prompt in `topic_research.py`
  - Remove rigid threshold rules:
    - ~~"STOP IMMEDIATELY if 3+ sources FROM 2+ DISTINCT DOMAINS AND ≥1 HIGH quality"~~
    - ~~"STOP if 3+ relevant sources from distinct domains"~~
    - ~~"ADEQUATE if 2+ sources but same domain"~~
    - ~~"CONTINUE if <2 relevant sources"~~
  - Replace with adaptive guidance:
    - "Assess whether the findings substantively answer the research question"
    - "Simple factual queries: 2-3 searches are usually sufficient"
    - "Comparative analysis: 4-6 searches to cover multiple perspectives"
    - "Complex multi-dimensional topics: use up to your budget limit"
    - "Stop when you are confident the findings address the research question, or when additional searches yield diminishing returns"
  - Preserve `urls_to_extract` recommendation guidance (unchanged)
- [ ] **5.2** Remove source-count/domain-count injection into reflection context
  - Current: reflection receives `sources_found: N, quality_distribution: {HIGH: X, ...}, distinct_domains: Y`
  - Change: the researcher can see its own accumulated sources; it doesn't need metadata pre-computed
  - Keep: total `tool_calls_used` / `max_tool_calls` budget visibility (so researcher knows its remaining budget)
- [ ] **5.3** Update early-exit heuristic in ReAct loop
  - Keep: budget hard cap (`max_tool_calls`) — always enforced
  - Keep: "no new sources found" exit — prevents infinite loops on exhausted topics
  - Remove: metadata-threshold early exit (3+ sources, 2+ domains, HIGH quality)
  - The LLM's own `continue_searching=False` / `research_complete=True` is the primary exit signal
- [ ] **5.4** Add tests for simplified reflection
  - Test: researcher continues beyond 3 sources on complex topics
  - Test: researcher stops early on simple topics (fewer than 3 searches)
  - Test: budget hard cap still enforced (max_tool_calls)
  - Test: "no new sources" exit still works
  - Test: `research_complete` signal from reflection terminates loop
  - Test: reflection rationale is substantive (not just threshold checks)
  - Test: no regression on research quality (sources are still relevant and diverse)
