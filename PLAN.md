# PLAN: Deep Research Architecture — Brief, Compression & Delegation

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24
**Reference:** `~/GitHub/open_deep_research`
**Status:** Draft
**Depends on:** All prior phases (think-tool, global compression, evaluation, autonomy — all complete)

---

## Context

Comparative analysis of foundry-mcp deep research against `open_deep_research` (RACE score 0.4344, #6 on Deep Research Bench) after completing two prior plan cycles (12 phases total) reveals four remaining structural gaps. Prior work aligned token management, compression, reflection patterns, and evaluation. What remains is:

1. **Query enrichment** — open_deep_research transforms raw user messages into a detailed research brief (source preferences, filled dimensions, specificity) before planning. foundry-mcp refines the query during planning but never creates a standalone research brief that drives all downstream phases.

2. **Compression timing** — open_deep_research compresses per-researcher findings *inside* the researcher subgraph before the supervisor sees them. foundry-mcp defers all compression to a separate phase after analysis, so supervision operates on raw source counts instead of actual content.

3. **Search depth** — open_deep_research allows 10 tool calls per researcher and 6 supervisor iterations (with 5 parallel researchers each). foundry-mcp caps at 5 searches per topic and 3 supervision rounds. Researchers also lack URL extraction capability (can search but can't deeply read specific pages).

4. **Supervision model** — open_deep_research's supervisor delegates *research tasks* (detailed paragraph-length directives) to parallel researcher agents via `ConductResearch`. foundry-mcp's supervisor generates follow-up *queries* (single-sentence strings). The delegation model produces more targeted, context-rich research assignments.

Phases are ordered by dependency chain: Phase 1 is independent with highest upstream impact. Phase 2 is prerequisite for Phase 4. Phase 4 depends on Phase 2 (supervisor needs compressed findings).

---

## Phase 1: Research Brief Generation

**Effort:** Low | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` (new)
- `src/foundry_mcp/core/research/models/deep_research.py` (state additions)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (phase wiring)
- `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py` (agent mapping)
- `src/foundry_mcp/config/research.py` (config fields)

### Problem

open_deep_research's `write_research_brief` step (prompts.py:44-77) transforms raw user messages into a structured `ResearchQuestion` with:
- Maximized specificity (unstated dimensions filled as open-ended rather than assumed)
- Source preferences (official/primary sources, peer-reviewed work, language-native sources)
- Explicit scope boundaries (what to include, what to exclude)

foundry-mcp's planning phase has a brief refinement sub-step (planning.py:80-114) that cleans up the query, but it's embedded within planning — not a dedicated phase. The refined brief is never stored as a distinct artifact and isn't available to downstream phases (supervision, synthesis) that could benefit from its specificity.

Poor query quality cascades: vague queries produce vague sub-queries which produce shallow research. The research brief is the single highest-leverage upstream quality amplifier.

### Design

Add a new BRIEF phase between CLARIFICATION and PLANNING:

1. **Phase insertion**: Add `BRIEF` to `DeepResearchPhase` enum after CLARIFICATION.
2. **Structured output**: LLM transforms raw query + clarification constraints into a `research_brief` string. The prompt adapts open_deep_research's approach: maximize specificity, prefer primary sources, fill unstated dimensions as open-ended, avoid unwarranted assumptions.
3. **State storage**: `state.research_brief` persists the brief for all downstream consumers.
4. **Planning integration**: `_execute_planning_async()` uses `state.research_brief` (when available) instead of `state.original_query` for sub-query decomposition. This replaces the existing inline refinement sub-step.
5. **Synthesis integration**: The brief is included in synthesis context so the report directly addresses the enriched question.
6. **Supervision integration**: Coverage assessment references the brief's scope boundaries.

The brief phase uses the research-tier model (not cheap/reflection) because query enrichment requires strong reasoning about unstated dimensions.

### Changes

1. Add `BRIEF` to `DeepResearchPhase` enum between `CLARIFICATION` and `PLANNING`
2. Add `research_brief: Optional[str] = None` to `DeepResearchState`
3. Create `phases/brief.py` with `BriefPhaseMixin`:
   - `_execute_brief_async()` — orchestrator
   - `_build_brief_system_prompt()` — adapted from open_deep_research's `transform_messages_into_research_topic_prompt`
   - `_build_brief_user_prompt()` — includes original query, clarification constraints, date
   - Uses `execute_structured_llm_call()` for parse-retry reliability
4. Add `BRIEFER` agent role to `PHASE_TO_AGENT` mapping in orchestration.py
5. Wire BRIEF phase into `workflow_execution.py` phase loop (after CLARIFICATION, before PLANNING)
6. Update `_execute_planning_async()`: when `state.research_brief` exists, use it as decomposition input instead of running inline refinement
7. Update `_build_synthesis_user_prompt()` to include research brief in context
8. Add config: `deep_research_brief_provider`, `deep_research_brief_model` (defaults to research-tier)
9. Add config flag: `deep_research_enable_brief: bool = True`
10. Add `phases/__init__.py` export

### Validation

- Test: brief enriches a vague query with specific dimensions and source preferences
- Test: brief preserves explicit user constraints from clarification phase
- Test: planning uses research_brief when available, falls back to original_query
- Test: brief skipped when config flag disabled
- Test: synthesis prompt includes research brief context
- Test: brief phase audit event emitted
- Test: non-fatal — if brief generation fails, planning uses original query

---

## Phase 2: Inline Per-Topic Compression Before Supervision

**Effort:** Medium | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (compression call)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` (refactor)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (content-aware assessment)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py` (orchestration)

### Problem

open_deep_research compresses per-researcher findings *inside* the researcher subgraph (`compress_research` node) before returning results to the supervisor. The supervisor thus operates on clean, compressed findings when deciding next steps — it reads actual research content, not just source counts.

foundry-mcp defers all per-topic compression to a separate COMPRESSION phase that runs *after* analysis. The supervision phase (between GATHERING and ANALYSIS) sees only raw metadata: source counts, quality scores, domain diversity. It cannot assess whether the *content* of findings actually covers the query's requirements.

This is the single largest architectural divergence from open_deep_research's supervision model. Coverage assessment without content visibility is fundamentally limited — the supervisor can't distinguish between 5 sources that all say the same thing and 5 sources that each cover a different dimension.

### Design

Move per-topic compression into the gathering phase. After each topic researcher's ReAct loop completes, immediately compress its findings:

1. **Inline compression**: At the end of `_execute_topic_research_async()`, after the ReAct loop and before returning `TopicResearchResult`, call the existing per-topic compression logic.
2. **Result enrichment**: `TopicResearchResult.compressed_findings` is populated inline (already exists as a field, currently populated later by separate compression phase).
3. **Supervision upgrade**: `_build_supervision_user_prompt()` now includes compressed findings per sub-query, not just source counts. The LLM can assess actual content coverage.
4. **Global compression adjustment**: The separate COMPRESSION phase (global cross-topic dedup) continues to run post-analysis, but operates on already-compressed per-topic findings instead of raw sources. Remove the per-topic compression from this phase (it's already done).
5. **Fallback**: If inline compression fails for a topic, supervision falls back to metadata-only assessment for that topic. Non-fatal.

### Changes

1. Extract `_compress_single_topic()` from `compression.py` into a reusable helper (or call it from `topic_research.py`)
2. Add compression call at end of `_execute_topic_research_async()` after ReAct loop completes
3. Populate `TopicResearchResult.compressed_findings` inline during gathering
4. Update `_build_supervision_user_prompt()` to include compressed findings per sub-query:
   - For each sub-query: include `compressed_findings[:2000]` (truncated for budget)
   - Shift prompt from "N sources found, M domains" to "N sources found, key findings: ..."
5. Update `_build_supervision_system_prompt()` to instruct LLM to assess content coverage, not just source diversity
6. Refactor `_execute_global_compression_async()` (COMPRESSION phase): remove per-topic compression loop, keep only cross-topic merge/dedup logic
7. Update `_execute_global_compression_async()` input: read from `TopicResearchResult.compressed_findings` instead of re-compressing raw sources
8. Add `deep_research_inline_compression: bool = True` config flag (enables per-topic compression during gathering)
9. Tests

### Validation

- Test: `TopicResearchResult.compressed_findings` is populated after gathering (not null)
- Test: supervision prompt includes compressed content excerpts
- Test: supervision coverage assessment references actual findings, not just counts
- Test: global compression phase operates on pre-compressed findings (no double compression)
- Test: inline compression failure is non-fatal (supervision falls back to metadata)
- Test: total workflow token usage doesn't significantly increase (compression happens same number of times, just earlier)
- Test: supervision produces more targeted follow-ups when it can see content

---

## Phase 3: Iteration Budget + Extract Tool for Researchers

**Effort:** Medium | **Impact:** Medium
**Files:**
- `src/foundry_mcp/config/research.py` (budget defaults)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (extract integration)
- `src/foundry_mcp/core/research/providers/tavily.py` (extract provider)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (iteration limits)

### Problem

**Budget gap**: open_deep_research allows 10 tool calls per researcher and 6 supervisor iterations with 5 parallel researchers each iteration. foundry-mcp caps at 5 searches per topic and 3 supervision rounds. For PhD-level research tasks (Deep Research Bench: 100 tasks across 22 fields), this limits exploration depth.

**Extract gap**: open_deep_research researchers can call `tavily_search` with `include_raw_content=True`, getting full page content alongside search snippets. foundry-mcp researchers only get search result snippets from providers. When a search reveals a promising URL, there's no way to deeply read that page's content. open_deep_research's `summarize_webpage()` processes raw content at ~50K chars, extracting structured summaries with key excerpts.

The extract tool is particularly valuable for: technical documentation (API specs, configuration guides), academic papers (methodology details, results tables), and comparison pages (feature matrices, pricing tables) — exactly the content types that Deep Research Bench evaluates.

### Design

**Budget uplift**:
- Raise `deep_research_topic_max_searches` default: 5 → 10 (matching open_deep_research's `max_react_tool_calls`)
- Raise `deep_research_max_supervision_rounds` default: 3 → 6 (matching `max_researcher_iterations`)
- The early-exit heuristic (Phase 5 of prior plan) already prevents over-searching — higher caps just remove artificial ceilings for complex queries.

**Extract tool integration**:
- Add a content extraction step to the per-topic ReAct loop. After search results return, the researcher can optionally extract full content from the most promising URLs.
- Use Tavily Extract API (already available via `tavily_extract` MCP tool) or direct HTTP fetch + LLM summarization.
- Extraction is budget-counted: each extract counts toward the per-topic tool call limit.
- The reflection step can recommend URLs for extraction (new field: `urls_to_extract`).
- Extracted content goes through the same source summarization pipeline (50K char cap, LLM summary).

### Changes

1. Update config defaults:
   - `deep_research_topic_max_searches`: 5 → 10
   - `deep_research_max_supervision_rounds`: 3 → 6
2. Rename `deep_research_topic_max_searches` to `deep_research_topic_max_tool_calls` (reflects that search + extract both count)
3. Add `_topic_extract()` method to `TopicResearchMixin`:
   - Input: list of URLs to extract
   - Uses Tavily Extract API or HTTP fetch
   - Summarizes extracted content via `SourceSummarizer`
   - Creates `ResearchSource` entries with full content
   - Respects concurrency semaphore
4. Update reflection decision schema: add `urls_to_extract: list[str]` field (optional, max 2 per iteration)
5. Update ReAct loop: after reflection, if `urls_to_extract` is non-empty, run extraction before next search
6. Update reflection system prompt: instruct LLM to recommend extraction when search snippet suggests rich content behind a URL
7. Add `deep_research_enable_extract: bool = True` config flag
8. Add `deep_research_extract_max_per_iteration: int = 2` config (caps extraction cost)
9. Tests

### Validation

- Test: higher iteration budget allows deeper research on complex queries
- Test: early-exit heuristic still fires (no regression on simple queries)
- Test: extract tool fetches and summarizes URL content
- Test: extracted sources are properly deduplicated against search sources
- Test: extraction failures are non-fatal (research continues with search sources)
- Test: reflection can recommend URLs for extraction
- Test: total tool calls (search + extract) respect `max_tool_calls` cap
- Test: supervision respects new round limit (6 vs 3)

---

## Phase 4: Supervisor Delegation Model

**Effort:** High | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (major refactor)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py` (delegation support)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (loop changes)
- `src/foundry_mcp/core/research/models/deep_research.py` (state additions)
- `src/foundry_mcp/config/research.py` (config additions)

### Problem

open_deep_research's supervisor delegates *research tasks* — detailed paragraph-length directives — via the `ConductResearch` tool. Each directive is a rich description of what to investigate, what perspective to take, and what evidence to seek. The supervisor can spawn up to 5 parallel researchers per iteration and run 6 iterations, seeing compressed findings between each round.

foundry-mcp's supervision generates *follow-up queries* — single-sentence strings appended to the sub-query list. The queries then go through the same gathering pipeline as the original queries. The supervisor cannot:
- Specify the research approach (compare vs. investigate vs. validate)
- Target specific gaps in existing findings
- Delegate to parallel researchers with distinct mandates
- See compressed findings between delegation rounds (Phase 2 fixes this prerequisite)

The delegation model is more powerful because it gives the supervisor *directive authority* over research strategy, not just query-generation authority.

### Design

Refactor supervision into a delegation-based model that mirrors open_deep_research's supervisor pattern:

1. **Delegation tool**: Define a `ResearchDirective` structured output (analogous to `ConductResearch`):
   ```
   research_topic: str  # Detailed paragraph-length directive
   perspective: str     # What angle to approach from
   evidence_needed: str # What specific evidence to seek
   priority: int        # 1=critical, 2=important, 3=nice-to-have
   ```

2. **Supervisor loop**: Instead of a single coverage-assessment LLM call:
   - Think step: analyze compressed findings, identify gaps (already exists from Phase 1 of prior plan)
   - Delegate step: generate 1-N `ResearchDirective` objects targeting specific gaps
   - Execute step: spawn parallel topic researchers for each directive
   - Compress step: inline compression of results (Phase 2)
   - Assess step: decide whether to continue or complete

3. **Parallel execution**: Directives execute as parallel topic researchers (reusing existing `_execute_topic_research_async()` infrastructure). Each directive becomes a new `SubQuery` with the directive's topic as its text.

4. **Think-tool enforcement**: Mandatory think step before delegation (articulate gaps) and after results return (assess coverage). Mirrors open_deep_research's "think before ConductResearch" pattern.

5. **Completion signal**: Supervisor can signal `ResearchComplete` when satisfied, or the iteration limit triggers automatic completion.

6. **Budget management**: `max_concurrent_research_units` config caps parallel researchers per delegation round. `max_supervision_rounds` caps total rounds.

### Changes

1. Add `ResearchDirective` dataclass to `models/deep_research.py`
2. Add `max_concurrent_research_units: int = 5` to `ResearchConfig`
3. Refactor `_execute_supervision_async()`:
   - Replace single LLM call with think → delegate → execute → assess loop
   - Think step produces gap analysis (existing)
   - Delegate step produces list of `ResearchDirective` objects (new)
   - Execute step spawns parallel topic researchers (reuse existing infrastructure)
   - Assess step evaluates results and decides continue/complete
4. Add `_build_delegation_prompt()` — system prompt for generating directives from gap analysis
5. Add `_parse_delegation_response()` — extract `ResearchDirective` objects from LLM response
6. Add `_execute_directives_async()` — spawn parallel researchers for directives, collect results
7. Update `workflow_execution.py`: supervision can now trigger inline gathering (not just follow-up query generation)
8. Add think-before-delegate and think-after-results calls (using reflection model)
9. Add `deep_research_delegation_model: bool = True` config flag (fall back to query-generation model when disabled)
10. Add `deep_research_max_concurrent_research_units: int = 5` config
11. Tests

### Validation

- Test: supervisor generates paragraph-length directives (not single-sentence queries)
- Test: directives target specific gaps identified in think output
- Test: parallel researcher execution respects concurrency limit
- Test: supervisor sees compressed findings from directive execution
- Test: think step runs before delegation and after results
- Test: completion signal terminates supervision loop
- Test: iteration limit triggers automatic completion
- Test: fallback to query-generation when delegation disabled
- Test: directive priorities influence execution order
- Test: total search budget respected across delegation rounds

