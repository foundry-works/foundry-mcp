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

- [x] **2.1** Add `_build_decomposition_critique_prompt()` to `PlanningPhaseMixin`
  - Takes generated sub-queries + original research brief as input
  - Evaluates: redundancies, missing perspectives, scope issues
  - Returns structured JSON: `{redundancies: [], gaps: [], adjustments: []}`
- [x] **2.2** Execute critique call after sub-query generation in `_execute_planning_async()`
  - Uses reflection role (cheap model)
  - Single round only (not iterative)
  - Runs before advancing to GATHERING phase
- [x] **2.3** Parse and apply critique adjustments to `state.sub_queries`
  - Merge identified redundancies
  - Add sub-queries for identified perspective gaps
  - Respect `max_sub_queries` bound after adjustments
- [x] **2.4** Add config flag `deep_research_enable_planning_critique: bool = True`
  - Update `ResearchConfig` in `src/foundry_mcp/config/research.py`
  - Add to `from_toml_dict()` parsing
  - Add to `DeepResearchConfig` sub-config dataclass
- [x] **2.5** Record critique in `state.metadata["planning_critique"]`
  - Store original sub-queries, critique response, and adjusted sub-queries
- [x] **2.6** Add tests for planning self-critique (28 tests)
  - Test: redundant sub-queries identified and merged
  - Test: missing perspectives added
  - Test: sub-query count respects bounds after adjustments
  - Test: critique skipped when config flag disabled
  - Test: no regression in existing tests (1940 passed)

---

## Phase 3: Global Note Compression Before Synthesis

- [x] **3.1** Add `COMPRESSION` to `DeepResearchPhase` enum
  - Insert between `ANALYSIS` and `SYNTHESIS`
  - Update `advance_phase()` logic in `DeepResearchState`
  - Update `PHASE_TO_AGENT` mapping in orchestration.py (added `COMPRESSOR` agent role)
- [x] **3.2** Add `_execute_global_compression_async()` to `CompressionMixin`
  - Takes all per-topic compressed findings + analysis findings/gaps/contradictions
  - Deduplicates cross-topic findings (same fact, different sources)
  - Merges related findings into coherent themes
  - Produces unified digest with consistent citation numbering
  - Flags cross-topic contradictions explicitly
- [x] **3.3** Add `compressed_digest` field to `DeepResearchState`
  - `compressed_digest: Optional[str] = None`
  - Populated by global compression, consumed by synthesis
- [x] **3.4** Update synthesis to use `compressed_digest` when available
  - In `_build_synthesis_user_prompt()`, prefer `compressed_digest` over raw findings
  - Fall back to raw findings when compression disabled or failed
  - Extracted `_build_synthesis_tail()` helper for shared source-reference + instructions
- [x] **3.5** Wire into workflow_execution.py phase loop
  - Add COMPRESSION phase handling between ANALYSIS and SYNTHESIS
  - Include phase-boundary reflection hook
  - Added TYPE_CHECKING stub for `_execute_global_compression_async`
- [x] **3.6** Add config flag `deep_research_enable_global_compression: bool = True`
  - Skip phase for single-topic research (no cross-topic dedup value)
  - Add provider/model/timeout config keys (`global_compression_provider`, `global_compression_model`, `global_compression_timeout`)
  - Added `global_compression` role to `_ROLE_RESOLUTION_CHAIN` (falls back to compression → research)
  - Added `compression` to `get_phase_timeout()`
  - Added `from_toml_dict()` parsing for all new config keys
- [x] **3.7** Add tests for global compression (29 tests)
  - Test: duplicate findings across topics are deduplicated
  - Test: cross-topic contradictions preserved and flagged
  - Test: citation numbering is consistent after merge (original numbers preserved)
  - Test: synthesis prompt size decreases with compression
  - Test: skipped for single-topic research
  - Test: graceful fallback when compression fails
  - Test: prompt includes all topics, analysis findings, contradictions
  - Test: config fields, TOML parsing, role resolution chain
  - Test: audit events emitted on success and failure
  - Test: no regression in existing tests (1969 passed)

---

## Phase 4: Research Quality Evaluation Framework

- [x] **4.1** Create evaluation package structure
  - `src/foundry_mcp/core/research/evaluation/__init__.py`
  - `src/foundry_mcp/core/research/evaluation/evaluator.py`
  - `src/foundry_mcp/core/research/evaluation/dimensions.py`
  - `src/foundry_mcp/core/research/evaluation/scoring.py`
- [x] **4.2** Define 6 evaluation dimensions with rubrics
  - Depth (1-5): thoroughness of investigation
  - Source Quality (1-5): credibility, diversity, recency
  - Analytical Rigor (1-5): reasoning quality, evidence use
  - Completeness (1-5): coverage of all query dimensions
  - Groundedness (1-5): claims supported by cited evidence
  - Structure (1-5): organization, readability, citations
- [x] **4.3** Implement LLM-as-judge evaluator
  - Takes: research query, final report, source list
  - Produces: per-dimension score + rationale + composite score (0-1)
  - Uses strong model (research-tier) via role-based resolution ("evaluation" role)
  - Handles structured output parsing with fallback
  - Low temperature (0.1) for scoring consistency
  - Integrates with `execute_llm_call()` lifecycle (heartbeat, audit, metrics, token-limit recovery)
- [x] **4.4** Add scoring normalization and composite calculation
  - Normalize each dimension to 0-1 via `(raw - 1) / 4`
  - Weighted composite (equal weights initially, configurable via `compute_composite(weights=)`)
  - Score variance across dimensions for confidence assessment
  - Clamping of out-of-range LLM scores to [1, 5]
- [x] **4.5** Add evaluation action to research action handler
  - `action="evaluate"` with `research_id` parameter in `core.py` dispatch
  - `_evaluate_research()` method in `ActionHandlersMixin`
  - `_handle_deep_research_evaluate()` MCP handler in `handlers_deep_research.py`
  - Registered as `deep-research-evaluate` action in router
  - Returns evaluation results in standard response envelope
- [x] **4.6** Add config keys for evaluation
  - `deep_research_evaluation_provider`
  - `deep_research_evaluation_model`
  - `deep_research_evaluation_timeout` (default 360s)
  - Added "evaluation" to `_ROLE_RESOLUTION_CHAIN` (falls back to research → analysis)
  - Updated `from_toml_dict()` parsing for all new config keys
  - Updated `DeepResearchConfig` sub-config with evaluation fields
- [x] **4.7** Store evaluation results in session metadata
  - `state.metadata["evaluation"]` with scores, rationales, composite, weights, variance
  - State persisted via `workflow.memory.save_deep_research()`
  - Audit events: `evaluation.started`, `evaluation.completed`, `evaluation.failed`
- [x] **4.8** Add test suite (70 tests)
  - Test: 6 dimensions defined with complete rubrics (scores 1-5)
  - Test: dimension names unique, frozen, and lookup works
  - Test: score normalization (1→0.0, 3→0.5, 5→1.0, out-of-range raises)
  - Test: composite normalizes to 0-1 for all input combinations
  - Test: dimensions produce independent scores (not all identical)
  - Test: poor reports score lower than comprehensive reports
  - Test: variance zero for uniform, nonzero for varied scores
  - Test: custom weights change composite
  - Test: prompt includes query, report, sources, all rubrics, JSON instruction
  - Test: long reports truncated, source count limited
  - Test: parsing handles code blocks, surrounding text, missing dimensions
  - Test: invalid JSON / missing scores key raises
  - Test: evaluation results persisted in session metadata
  - Test: audit events emitted (started, completed)
  - Test: LLM failure and parse failure return error WorkflowResult
  - Test: config fields, TOML parsing, role resolution chain
  - Test: action handler validation (missing research_id, missing report, not found)
  - Test: no regression in existing tests (2039 passed)

---

## Phase 5: Enhanced Per-Researcher Tool Autonomy

### Phase 5.1: Increase Default Loop Depth

- [x] **5.1.1** Update `deep_research_topic_max_searches` default from 3 to 5
  - In `src/foundry_mcp/config/research.py`, `research_sub_configs.py`, `gathering.py`
- [x] **5.1.2** Add cost-aware early-exit heuristic in ReAct loop
  - Added `_check_early_exit()` method to `TopicResearchMixin`
  - Condition: `sources_found >= 3 AND distinct_domains >= 2 AND quality_HIGH >= 1`
  - Triggers before reflection to save both reflection and search costs
- [x] **5.1.3** Update reflection prompt for aggressive early stopping
  - Added "STOP IMMEDIATELY" rule for high-quality diverse sources
  - Added "diminishing returns" guidance
  - Reordered decision rules by priority

### Phase 5.2: Think-Tool Step Within ReAct Loop

- [x] **5.2.1** Add `_topic_think()` method to `TopicResearchMixin`
  - Articulates: what was found, what angle to try next, why it matters
  - Uses reflection model (cheap) via role-based resolution
  - Returns structured JSON: `{reasoning, next_query, tokens_used}`
- [x] **5.2.2** Integrate into ReAct loop between reflect and next search
  - After reflection decides `continue_searching=true`
  - Think output's `next_query` preferred over reflection's `refined_query`
  - Record think output in `TopicResearchResult.reflection_notes` with `[think]` prefix
- [x] **5.2.3** Add tests for within-loop think step (8 tests)
  - Test: think output produces actionable query rationale
  - Test: refined queries reflect think-step reasoning (preferred over reflection)
  - Test: failure handled gracefully (no crash)
  - Test: prompt includes full context
  - Test: integration in ReAct loop sequence

### Phase 5.3: Cross-Researcher Content Deduplication

- [x] **5.3.1** Add `content_similarity()` helper to `_helpers.py`
  - Character n-gram (shingling) Jaccard similarity
  - Returns similarity score 0-1
  - Includes length-ratio pre-check for efficiency
  - `_normalize_content_for_dedup()` strips whitespace and copyright boilerplate
- [x] **5.3.2** Extend dedup logic in `_topic_search()` for content similarity
  - After URL/title dedup, check content similarity for remaining sources
  - Mark similar-content sources as duplicates (skip adding to state)
  - Log dedup events at DEBUG level for observability
  - Skip check for short content (< 100 chars)
- [x] **5.3.3** Add config flags
  - `deep_research_enable_content_dedup: bool = True`
  - `deep_research_content_dedup_threshold: float = 0.8`
  - Updated `from_toml_dict()`, `DeepResearchConfig`, `deep_research_config` property
- [x] **5.3.4** Add tests for content deduplication (10 tests)
  - Test: mirror/syndicated content from different URLs is deduplicated
  - Test: genuinely different content from similar titles is preserved
  - Test: dedup disabled when config flag off
  - Test: short content bypasses dedup
  - Test: identical texts return 1.0, different texts < 0.3
  - Test: empty texts return 0.0
  - Test: config keys, TOML parsing, sub-config fields
  - Test: no regression in existing tests (2067 passed)

---

## Sign-off

- [x] All phases reviewed and approved
- [x] Tests pass: `pytest tests/core/research/ -x` (2067 passed, 6 skipped)
- [x] No regressions in existing tests
- [ ] Code review completed
