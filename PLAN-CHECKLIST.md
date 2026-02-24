# PLAN-CHECKLIST: Deep Research Quality — Think-Tool, Compression & Evaluation

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23

---

## Phase 1: Think-Tool Deliberation in Supervision

- [x] **1.1** Add `_build_think_prompt()` to `SupervisionPhaseMixin`
  - Generates gap-analysis-only prompt from per-sub-query coverage data
  - Does NOT produce follow-up queries — only articulates what's found and what's missing
  - Uses coverage_data dict already built by `_build_coverage_data()`
- [x] **1.2** Execute think call before coverage assessment in `_execute_supervision_async()`
  - Separate LLM call using reflection role (cheap model)
  - Guarded by heuristic fast-path: skip when `supervision_round > 0` and coverage sufficient
  - Timeout matches `deep_research_reflection_timeout`
- [x] **1.3** Pass think output into `_build_supervision_user_prompt()`
  - Think output becomes a `<gap_analysis>` section in the follow-up generation prompt
  - Follow-up queries should reference specific gaps identified in think output
- [x] **1.4** Record think output in `state.metadata["supervision_history"]`
  - Each supervision round entry gets a `think_output` field
  - Preserves full deliberation chain for traceability
- [x] **1.5** Add tests for think-tool integration
  - Test: think output contains per-sub-query gap analysis
  - Test: follow-up queries reference gaps from think output
  - Test: think step skipped on heuristic fast-path
  - Test: token cost bounded (cheap model, ~500 tokens per round)

---

## Phase 2: Think-Tool Self-Critique at Planning Boundary

- [ ] **2.1** Add `_build_decomposition_critique_prompt()` to `PlanningPhaseMixin`
  - Takes generated sub-queries + original research brief as input
  - Evaluates: redundancies, missing perspectives, scope issues
  - Returns structured JSON: `{redundancies: [], gaps: [], adjustments: []}`
- [ ] **2.2** Execute critique call after sub-query generation in `_execute_planning_async()`
  - Uses reflection role (cheap model)
  - Single round only (not iterative)
  - Runs before advancing to GATHERING phase
- [ ] **2.3** Parse and apply critique adjustments to `state.sub_queries`
  - Merge identified redundancies
  - Add sub-queries for identified perspective gaps
  - Respect `max_sub_queries` bound after adjustments
- [ ] **2.4** Add config flag `deep_research_enable_planning_critique: bool = True`
  - Update `ResearchConfig` in `src/foundry_mcp/config/research.py`
  - Add to `from_dict()` parsing
- [ ] **2.5** Record critique in `state.metadata["planning_critique"]`
  - Store original sub-queries, critique response, and adjusted sub-queries
- [ ] **2.6** Add tests for planning self-critique
  - Test: redundant sub-queries identified and merged
  - Test: missing perspectives added
  - Test: sub-query count respects bounds after adjustments
  - Test: critique skipped when config flag disabled
  - Test: no regression in existing planning tests

---

## Phase 3: Global Note Compression Before Synthesis

- [ ] **3.1** Add `COMPRESSION` to `DeepResearchPhase` enum
  - Insert between `ANALYSIS` and `SYNTHESIS`
  - Update `advance_phase()` logic in `DeepResearchState`
  - Update `PHASE_TO_AGENT` mapping in orchestration.py
- [ ] **3.2** Add `_execute_global_compression_async()` to `CompressionMixin`
  - Takes all per-topic compressed findings + analysis findings/gaps/contradictions
  - Deduplicates cross-topic findings (same fact, different sources)
  - Merges related findings into coherent themes
  - Produces unified digest with consistent citation numbering
  - Flags cross-topic contradictions explicitly
- [ ] **3.3** Add `compressed_digest` field to `DeepResearchState`
  - `compressed_digest: Optional[str] = None`
  - Populated by global compression, consumed by synthesis
- [ ] **3.4** Update synthesis to use `compressed_digest` when available
  - In `_build_synthesis_user_prompt()`, prefer `compressed_digest` over raw findings
  - Fall back to raw findings when compression disabled or failed
- [ ] **3.5** Wire into workflow_execution.py phase loop
  - Add COMPRESSION phase handling between ANALYSIS and SYNTHESIS
  - Include phase-boundary reflection hook
- [ ] **3.6** Add config flag `deep_research_enable_global_compression: bool = True`
  - Skip phase for single-topic research (no cross-topic dedup value)
  - Add provider/model config keys
- [ ] **3.7** Add tests for global compression
  - Test: duplicate findings across topics are deduplicated
  - Test: cross-topic contradictions preserved and flagged
  - Test: citation numbering is consistent after merge
  - Test: synthesis prompt size decreases with compression
  - Test: skipped for single-topic research
  - Test: graceful fallback when compression fails

---

## Phase 4: Research Quality Evaluation Framework

- [ ] **4.1** Create evaluation package structure
  - `src/foundry_mcp/core/research/evaluation/__init__.py`
  - `src/foundry_mcp/core/research/evaluation/evaluator.py`
  - `src/foundry_mcp/core/research/evaluation/dimensions.py`
  - `src/foundry_mcp/core/research/evaluation/scoring.py`
- [ ] **4.2** Define 6 evaluation dimensions with rubrics
  - Depth (1-5): thoroughness of investigation
  - Source Quality (1-5): credibility, diversity, recency
  - Analytical Rigor (1-5): reasoning quality, evidence use
  - Completeness (1-5): coverage of all query dimensions
  - Groundedness (1-5): claims supported by cited evidence
  - Structure (1-5): organization, readability, citations
- [ ] **4.3** Implement LLM-as-judge evaluator
  - Takes: research query, final report, source list
  - Produces: per-dimension score + rationale + composite score (0-1)
  - Uses strong model (research-tier) for evaluation accuracy
  - Handles structured output parsing with fallback
- [ ] **4.4** Add scoring normalization and composite calculation
  - Normalize each dimension to 0-1
  - Weighted composite (equal weights initially, configurable later)
  - Confidence interval based on score variance across dimensions
- [ ] **4.5** Add evaluation action to research action handler
  - `action="evaluate"` with `research_id` parameter
  - Loads completed research report from session
  - Returns evaluation results in standard response envelope
- [ ] **4.6** Add config keys for evaluation
  - `deep_research_evaluation_provider`
  - `deep_research_evaluation_model`
  - Update `ResearchConfig` and `from_dict()`
- [ ] **4.7** Store evaluation results in session metadata
  - `state.metadata["evaluation"]` with scores, rationales, composite
- [ ] **4.8** Add test suite
  - Test: consistent scores for identical reports (low variance)
  - Test: poor reports score lower than comprehensive reports
  - Test: dimensions produce independent scores
  - Test: composite normalizes to 0-1
  - Test: evaluation results persisted in session metadata
  - Test: action handler returns evaluation in response envelope

---

## Phase 5: Enhanced Per-Researcher Tool Autonomy

### Phase 5.1: Increase Default Loop Depth

- [ ] **5.1.1** Update `deep_research_topic_max_searches` default from 3 to 5
  - In `src/foundry_mcp/config/research.py`
- [ ] **5.1.2** Add cost-aware early-exit heuristic in ReAct loop
  - In `topic_research.py` ReAct loop (before reflection call)
  - Condition: `sources_found >= 3 AND distinct_domains >= 2 AND quality_HIGH >= 1`
  - Sets `research_complete=True`, skips remaining iterations
- [ ] **5.1.3** Update reflection prompt for aggressive early stopping
  - Strengthen "STOP" rules when quality is high
  - Reduce bias toward continuing when coverage is already good

### Phase 5.2: Think-Tool Step Within ReAct Loop

- [ ] **5.2.1** Add `_topic_think()` method to `TopicResearchMixin`
  - Articulates: what was found, what angle to try next, why it matters
  - Uses reflection model (cheap)
  - Returns structured reasoning for next-query construction
- [ ] **5.2.2** Integrate into ReAct loop between reflect and next search
  - After reflection decides `continue_searching=true`
  - Think output feeds into query refinement
  - Record think output in `TopicResearchResult.reflection_notes`
- [ ] **5.2.3** Add tests for within-loop think step
  - Test: think output produces actionable query rationale
  - Test: refined queries reflect think-step reasoning
  - Test: token cost bounded

### Phase 5.3: Cross-Researcher Content Deduplication

- [ ] **5.3.1** Add `_content_similarity_hash()` helper to `_helpers.py`
  - Character n-gram overlap or simhash-based approach
  - Returns similarity score 0-1
  - Threshold: >0.8 = duplicate
- [ ] **5.3.2** Extend dedup logic in `_topic_search()` for content similarity
  - After URL/title dedup, check content similarity for remaining sources
  - Mark similar-content sources as duplicates
  - Log dedup events for observability
- [ ] **5.3.3** Add config flag `deep_research_enable_content_dedup: bool = True`
- [ ] **5.3.4** Add tests for content deduplication
  - Test: mirror/syndicated content from different URLs is deduplicated
  - Test: genuinely different content from similar titles is preserved
  - Test: dedup disabled when config flag off

---

## Sign-off

- [ ] All phases reviewed and approved
- [ ] Tests pass: `pytest tests/core/research/ -x`
- [ ] No regressions in existing tests
- [ ] Code review completed
