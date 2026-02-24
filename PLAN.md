# PLAN: Deep Research Quality — Think-Tool, Compression & Evaluation

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23
**Reference:** `~/GitHub/open_deep_research`
**Status:** Draft
**Depends on:** Phases 1-6 of prior plan (all complete)

---

## Context

Comparative analysis of foundry-mcp deep research against `open_deep_research` (RACE score 0.4344, #6 on Deep Research Bench) identified five remaining alignment gaps after completing prior Phases 1-6 (token limits, brief refinement, synthesis prompts, structured truncation, reflection enforcement, supervision architecture).

The completed work gave us the structural bones — supervision phase, per-topic ReAct loops, fetch-time summarization, phase-boundary reflection. What's missing is the **deliberation quality** (think-tool pattern), **information density management** (global compression), and **measurement** (evaluation framework) that distinguish high-performing deep research systems.

Phases are ordered by impact-to-effort ratio. Phases 1-2 are low-effort prompt/flow changes. Phases 3-4 are medium-effort architectural additions. Phase 5 is a larger effort that builds on measurement from Phase 4.

---

## Phase 1: Think-Tool Deliberation in Supervision

**Effort:** Low | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (lines 86-168)
- `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py` (lines 526-631)

### Problem

open_deep_research's highest-impact quality pattern is the `think_tool` — a no-op tool that forces the LLM to explicitly reason through findings before acting. The supervisor uses it before every `ConductResearch` delegation and after every result return (prompts.py:79-136). The key constraint: think_tool must NOT run in parallel with action tools, creating deliberate sequential reasoning pauses.

foundry-mcp's supervision phase (`_execute_supervision_async`) makes a single LLM call for coverage assessment (line 150-168). There's no structured deliberation step where the model articulates *what it found*, *what's missing*, and *why specific follow-ups would fill gaps* before generating follow-up queries.

The `async_think_pause` in orchestration.py is a phase-boundary reflection (runs after a phase completes). It's not the same as within-phase deliberation that shapes the phase's own outputs.

### Design

Add a two-step deliberation pattern to supervision's coverage assessment:

1. **Think step**: Ask the LLM to analyze coverage gaps without producing follow-up queries. Force it to articulate: what was found per sub-query, what domains are represented, what perspectives are missing, and what specific information gaps exist. This is the "think_tool" equivalent.

2. **Act step**: Feed the think output into the existing follow-up query generation prompt. The LLM now generates follow-ups grounded in explicit gap analysis rather than making both assessments simultaneously.

This mirrors open_deep_research's pattern (think before ConductResearch, think after results) but adapted to foundry-mcp's single-prompt architecture. The think step is a separate LLM call, not a tool call, since we don't use tool-calling loops.

### Changes

1. Add `_build_think_prompt()` to `SupervisionPhaseMixin` — generates a gap-analysis-only prompt from coverage data
2. Execute think call before the existing coverage assessment call in `_execute_supervision_async()`
3. Pass think output as context into `_build_supervision_user_prompt()` so follow-up generation is grounded
4. Record think output in `state.metadata["supervision_history"]` for traceability
5. Use cheap model (reflection role) for the think step — this is fast analytical reasoning, not synthesis
6. Guard: skip think step when heuristic fast-path triggers (round > 0, coverage sufficient)

### Validation

- Test: think output contains per-sub-query gap analysis when coverage is incomplete
- Test: follow-up queries reference specific gaps identified in think output
- Test: think step is skipped when heuristic fast-path triggers
- Test: supervision with think-tool produces more targeted follow-ups than without (qualitative)
- Test: total token cost increase is bounded (think step uses cheap model, adds ~500 tokens per round)

---

## Phase 2: Think-Tool Self-Critique at Planning Boundary

**Effort:** Low | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py` (lines 49-309)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (lines 229-280)

### Problem

open_deep_research's supervisor thinks before every research delegation — evaluating whether sub-queries cover the research space, identifying redundancies, and ensuring diverse perspectives (prompts.py:86-92). foundry-mcp's planning phase generates sub-queries in a single pass (planning.py `_execute_planning_async`). The research brief is refined (Phase 2 of prior plan), but sub-query decomposition isn't self-critiqued before expensive gathering begins.

Poor decomposition cascades: redundant sub-queries waste search budget, missing perspectives leave gaps that supervision must later fill with follow-up queries (more expensive than getting it right upfront).

### Design

After sub-query generation, add a self-critique step that validates the decomposition before advancing to GATHERING:

1. Present the generated sub-queries back to an LLM with the original research brief
2. Ask it to evaluate: Are there redundancies? Missing perspectives? Overly broad queries? Overly narrow queries?
3. If critique identifies issues, apply adjustments (merge redundant queries, add missing perspectives, refine scope)
4. Limit to one critique round (not iterative) to bound cost

This is NOT the same as the existing phase-boundary reflection (which assesses "should we proceed?" generically). This is a targeted self-critique of the decomposition quality specifically.

### Changes

1. Add `_build_decomposition_critique_prompt()` to `PlanningPhaseMixin`
2. After sub-query generation in `_execute_planning_async()`, execute critique LLM call
3. Parse critique response for: redundancies to merge, gaps to fill, scope adjustments
4. Apply adjustments to `state.sub_queries` before returning
5. Use cheap model (reflection role) — this is analytical, not creative
6. Add config flag `deep_research_enable_planning_critique: bool = True` to allow disabling
7. Record critique in `state.metadata["planning_critique"]` for observability

### Validation

- Test: redundant sub-queries are identified and merged (e.g., "AI safety regulations" + "AI governance policies")
- Test: missing perspectives are added (e.g., historical, economic angles for a policy question)
- Test: critique does not run when disabled via config flag
- Test: total sub-query count stays within configured bounds after critique adjustments
- Test: existing planning tests pass without regression

---

## Phase 3: Global Note Compression Before Synthesis

**Effort:** Medium | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` (lines 47-336)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (lines 124-616)
- `src/foundry_mcp/core/research/models/deep_research.py` (state model)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

### Problem

open_deep_research has a two-level compression pipeline:
1. **Per-researcher compression** (deep_researcher.py:511) — each researcher's messages are compressed into structured notes after their ReAct loop
2. **Global note accumulation** (deep_researcher.py:323-330) — all researcher notes are concatenated and fed to final report generation

foundry-mcp has per-topic compression (compression.py) but no equivalent global compression pass. The synthesis phase receives the full analysis output (structured findings, sources, contradictions, gaps) directly. For complex multi-topic research, this means synthesis receives a large, heterogeneous prompt that may exceed context limits or produce incoherent reports because the model must simultaneously comprehend, deduplicate, and synthesize across topics.

### Design

Add a global compression step between analysis and synthesis that:

1. Takes all per-topic compressed findings + analysis output (findings, gaps, contradictions)
2. Deduplicates cross-topic findings (same fact from different sources)
3. Merges related findings into coherent themes
4. Produces a unified research digest with consistent citation numbering
5. Flags cross-topic contradictions explicitly

This is NOT the same as per-topic compression (which preserves everything verbatim). Global compression actively deduplicates and synthesizes across topics while preserving all unique information.

### Changes

1. Add `_execute_global_compression_async()` to `CompressionMixin`
2. Add `COMPRESSION` phase to `DeepResearchPhase` enum (between ANALYSIS and SYNTHESIS)
3. Build prompt that receives all topic findings grouped by theme
4. Store result in `state.compressed_digest: Optional[str]` (new field)
5. Update synthesis to use `compressed_digest` when available instead of raw findings
6. Add config: `deep_research_enable_global_compression: bool = True`
7. Wire into workflow_execution.py phase loop
8. Use research-tier model (this is substantive synthesis, not cheap reflection)

### Validation

- Test: global compression deduplicates identical findings across topics
- Test: cross-topic contradictions are preserved and flagged
- Test: citation numbering is consistent across merged topics
- Test: synthesis prompt size decreases after global compression
- Test: synthesis quality does not degrade (compressed digest preserves all unique information)
- Test: phase is skipped when single-topic research (no cross-topic value)

---

## Phase 4: Research Quality Evaluation Framework

**Effort:** Medium | **Impact:** Medium
**Files:**
- New: `src/foundry_mcp/core/research/evaluation/` (new package)
- New: `tests/core/research/evaluation/`

### Problem

open_deep_research includes a comprehensive evaluation framework (evaluators.py) that scores research output across 11 dimensions using LLM-as-judge:
- Research Depth, Source Quality, Analytical Rigor, Practical Value, Balance & Objectivity
- Writing Quality, Relevance, Structure, Correctness, Groundedness, Completeness

Each dimension is scored 1-5, normalized to 0-1, producing a composite RACE score. This enables data-driven decisions about whether architectural changes (like our supervision phase) actually improve output quality.

foundry-mcp has no evaluation framework. We can't measure whether Phases 1-3 of this plan improve research quality, and we can't benchmark against open_deep_research's 0.4344 RACE score.

### Design

Build an evaluation module that can score completed research reports. Design principles:

1. **Dimension-based scoring** — evaluate independently on 6 core dimensions (subset of open_deep_research's 11, focused on the most discriminating):
   - **Depth**: thoroughness of investigation
   - **Source Quality**: credibility, diversity, recency of sources
   - **Analytical Rigor**: quality of reasoning, evidence use
   - **Completeness**: coverage of all query dimensions
   - **Groundedness**: claims supported by cited evidence
   - **Structure**: organization, readability, citation consistency

2. **LLM-as-judge** — use a strong model to evaluate against rubrics
3. **Batch evaluation** — can score multiple reports for A/B testing
4. **Integration with deep research action** — optional `evaluate=True` flag on `deep-research-report` action

### Changes

1. Create `src/foundry_mcp/core/research/evaluation/__init__.py`
2. Create `src/foundry_mcp/core/research/evaluation/evaluator.py` — core evaluation logic
3. Create `src/foundry_mcp/core/research/evaluation/dimensions.py` — rubric definitions per dimension
4. Create `src/foundry_mcp/core/research/evaluation/scoring.py` — score normalization, composite calculation
5. Add evaluation action to research action handler: `action="evaluate"` with `research_id` parameter
6. Add config: `deep_research_evaluation_provider`, `deep_research_evaluation_model`
7. Store evaluation results in research session metadata
8. Create test suite with fixture reports of known quality

### Validation

- Test: evaluator produces consistent scores for identical reports (low variance across runs)
- Test: obviously poor reports score lower than comprehensive reports
- Test: each dimension produces independent scores (not all identical)
- Test: evaluation results are persisted in session metadata
- Test: composite score normalizes correctly to 0-1 range

---

## Phase 5: Enhanced Per-Researcher Tool Autonomy

**Effort:** High | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (lines 66-237)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py` (lines 430-570)
- `src/foundry_mcp/config/research.py`

### Problem

open_deep_research gives each researcher a full tool-calling ReAct loop with `max_react_tool_calls=10` — researchers can search, read results, think about gaps, search differently, and iterate up to 10 times per topic. foundry-mcp's topic agents have `max_searches=3` and a simpler loop: search → reflect → (maybe refine query) → search.

The key differences:
1. **Loop depth**: 3 vs 10 max iterations (open_deep_research researchers can dig deeper)
2. **Tool diversity**: open_deep_research researchers can use MCP tools, native web search, and Tavily; foundry-mcp researchers only use configured search providers
3. **Think-tool integration**: open_deep_research researchers use think_tool between every search; foundry-mcp researchers use structured reflection (similar but not as free-form)
4. **Concurrency model**: open_deep_research runs up to 5 researchers in parallel; foundry-mcp uses semaphore-bounded concurrency (configurable)

### Design

Enhance per-topic researcher autonomy in three sub-phases:

**5.1: Increase default loop depth**
- Change `deep_research_topic_max_searches` default from 3 to 5
- Add a cost-aware early exit: if sources are high-quality and diverse after 2 iterations, stop regardless of max
- The current reflection decision rules already support this but the hard cap is too low

**5.2: Add think-tool step within ReAct loop**
- After each search+reflect cycle, add a brief think step before the next search
- Think step articulates: what was found, what angle to try next, why that angle matters
- This grounds query refinement in explicit reasoning (currently refinement comes from reflection's `refined_query` field which is often generic)

**5.3: Cross-researcher deduplication improvement**
- Currently dedup uses URL + normalized title matching via shared `seen_urls` / `seen_titles` sets
- Add content-similarity dedup: if two sources from different URLs have >80% content overlap (via fuzzy hash), mark as duplicate
- This prevents wasted synthesis effort on mirror/syndicated content

### Changes

**5.1:**
1. Update default `deep_research_topic_max_searches` from 3 to 5
2. Add early-exit heuristic in ReAct loop: if `sources_found >= 3` AND `distinct_domains >= 2` AND `quality_distribution["HIGH"] >= 1`, set `research_complete=True` regardless of iteration count
3. Update reflection prompt to be more aggressive about early stopping when quality is high

**5.2:**
4. Add `_topic_think()` method to `TopicResearchMixin`
5. Call between reflect and next search iteration in the ReAct loop
6. Think output feeds into next search query construction
7. Use reflection model (cheap) for think step

**5.3:**
8. Add `_content_similarity_hash()` helper to `_helpers.py` using simhash or character n-gram overlap
9. Extend dedup logic in `_topic_search()` to check content similarity for sources with different URLs
10. Add config: `deep_research_enable_content_dedup: bool = True`

### Validation

- Test: researchers with max_searches=5 find more diverse sources than max_searches=3 on complex topics
- Test: early-exit heuristic triggers when high-quality diverse sources are found quickly
- Test: think step produces actionable next-query rationale
- Test: content-similar sources from different URLs are deduplicated
- Test: total token cost increase is bounded (think steps use cheap model, early-exit reduces unnecessary iterations)
- Test: no regression in existing topic research tests

---

## Reference: Alignment Status

| Area | Status | Phase |
|------|--------|-------|
| Model token limits | Aligned | Prior Phase 1 |
| Research brief refinement | Aligned | Prior Phase 2 |
| Synthesis prompt engineering | Aligned | Prior Phase 3 |
| Structured token recovery | Aligned | Prior Phase 4 |
| Topic reflection enforcement | Aligned | Prior Phase 5 |
| Supervision architecture | Aligned | Prior Phase 6 |
| Fetch-time summarization | Aligned | Prior work |
| Per-topic compression | Aligned | Prior work |
| Per-topic ReAct loops | Aligned | Prior work |
| Multi-model cost routing | Aligned | Prior work |
| **Think-tool deliberation** | **Gap** | **This plan Phase 1** |
| **Planning self-critique** | **Gap** | **This plan Phase 2** |
| **Global compression** | **Gap** | **This plan Phase 3** |
| **Evaluation framework** | **Gap** | **This plan Phase 4** |
| **Researcher tool autonomy** | **Partial** | **This plan Phase 5** |
