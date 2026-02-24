# PLAN-CHECKLIST: Deep Research Alignment with open_deep_research

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23

---

## Phase 1: Update Model Token Limits

- [x] **1.1** Update `model_token_limits.json` with accurate upstream values
  - GPT-4.1 variants: 1,000,000 → 1,047,576
  - Add gpt-4.1-nano (1,047,576), gpt-4o (128,000), gpt-4o-mini (128,000)
  - Add gemini-2.5-pro (1,048,576), gemini-2.5-flash (1,048,576)
  - Add gemini-1.5-pro (2,097,152)
- [x] **1.2** Update `_lifecycle.py` inline `MODEL_TOKEN_LIMITS` fallback dict to match json
- [x] **1.3** Verify `truncate_to_token_estimate` loads from json, not just hardcoded dict
- [x] **1.4** Run existing token truncation tests — confirm no regressions (66/66 passed)

---

## Phase 2: Research Brief Generation

- [x] **2.1** Add `_build_brief_refinement_prompt()` to `planning.py`
  - Original prompt (not copied), modeled after upstream principles
  - Maximize specificity, fill unstated dimensions as open-ended
  - Prefer primary sources, preserve language preference
- [x] **2.2** Add brief-refinement LLM call at start of `_execute_planning_async()`
  - Execute before sub-query decomposition
  - Use cheap model role (summarization or new "brief" role)
- [x] **2.3** Store refined brief in `state.research_brief`
- [x] **2.4** Wire refined brief as input to `_build_planning_user_prompt()`
- [x] **2.5** Add "brief" to `_ROLE_RESOLUTION_CHAIN` in `research.py` if using new role
  - N/A: used existing "summarization" role instead of a new "brief" role
- [x] **2.6** Test: ambiguous query produces more specific brief
- [x] **2.7** Test: sub-queries are grounded in refined brief
- [x] **2.8** Test: existing planning tests pass (additive change) — 2015 passed, 0 failed

---

## Phase 3: Synthesis Prompt Engineering

- [x] **3.1** Add language detection directive to `_build_synthesis_system_prompt()`
  - Detect language from user query, instruct report in same language
- [x] **3.2** Add structure-adaptive directives
  - Comparison → side-by-side structure (Comparative Analysis, Overview of [Subject A/B])
  - Enumeration → item-per-section or single-section list
  - Explanation → overview + Key Findings with Theme/Category subsections + Conclusions
  - How-to → Prerequisites + Step 1..N + Conclusions
- [x] **3.3** Add anti-pattern guardrails
  - No meta-commentary ("based on the research", "the findings show")
  - No hedging openers ("it appears that", "it seems")
  - No self-reference ("as an AI", "I found that")
- [x] **3.4** Enforce citation format: inline `[N]` + auto-appended numbered source section
  - System prompt instructs inline [N] citations; Sources section auto-appended by `_citation_postprocess.py`
- [x] **3.5** Add query-type hint to `_build_synthesis_user_prompt()`
  - `_classify_query_type()` classifies query → comparison/enumeration/howto/explanation
  - "Query type hint" line added to Instructions section of user prompt
- [x] **3.6** Test: non-English query produces non-English report directive
  - `TestLanguageDirective` (3 tests): language directive present, Chinese example, Chinese query builds prompt
- [x] **3.7** Test: comparison query produces comparison structure
  - `TestStructureAdaptive` (5 tests): each query type gets correct structure, all include Conclusions
- [x] **3.8** Test: output is free of anti-pattern phrases
  - `TestAntiPatternGuardrails` (4 tests): meta-commentary, hedging, self-reference, direct writing
- [x] **3.9** Test: citations are consistently formatted
  - `TestCitationFormat` (3 tests): inline [N], no Sources section, Conflicting Information preserved
  - 1841 passed, 6 skipped, 0 failures across full research test suite

---

## Phase 4: Message-Aware Token Recovery

- [x] **4.1** Add `_structured_truncation()` helper to `_helpers.py`
  - `_split_prompt_sections()` splits at markdown header boundaries
  - `structured_truncate_blocks()` truncates longest sections first, preserving protected sections
  - Source entry detection via `_SOURCE_ENTRY_PATTERNS` for synthesis, analysis, compression formats
- [x] **4.2** Add quality-aware source dropping
  - `structured_drop_sources()` scores by `[high]`/`[medium]`/`[low]` markers
  - Falls back to length-based scoring within same quality tier
  - Drops lowest-quality, largest sources first
- [x] **4.3** Update retry loop in `execute_llm_call()` (lines 358-420)
  - `_apply_truncation_strategy()` dispatches based on retry count
  - Retry 1: `structured_truncate_blocks` (longest blocks first)
  - Retry 2: `structured_drop_sources` (lowest quality or longest)
  - Retry 3: `truncate_to_token_estimate` char-based truncation (fallback)
  - Each strategy falls back to char-based when no structure is found
- [x] **4.4** Test: structured truncation preserves high-quality sources
  - `TestStructuredTruncateBlocks` (6 tests) + `TestStructuredTruncationIntegration.test_structured_prompt_preserves_high_quality_on_retry`
- [x] **4.5** Test: char-based fallback still works when structured fails
  - `TestStructuredTruncateBlocks.test_fallback_for_unstructured_prompt` + `TestStructuredTruncationIntegration.test_char_fallback_works_after_structured_fails`
- [x] **4.6** Test: no regression in existing token recovery tests
  - 1867 passed, 6 skipped, 0 failures across full research test suite

---

## Phase 5: Reflection Enforcement

- [x] **5.1** Update reflection system prompt in `_topic_reflect()` (lines 395-417)
  - Hard stop: research_complete=true when 3+ sources from distinct domains
  - Continue: only if <2 relevant sources found
  - Rationale must articulate specific gap to be filled
  - Added domain diversity counting (with www. dedup) and `Distinct source domains: N` to user prompt
  - Numbered decision rules: STOP, CONTINUE, ADEQUATE, NO RESULTS, rationale requirement
- [x] **5.2** Add guard: force research_complete if iteration >= max_searches
  - Regardless of reflection LLM decision
  - Added explicit `logger.info` when max_searches cap is hit
- [x] **5.3** Add logging of reflection rationale for observability
  - Every reflection decision now logged with `logger.info` showing complete/continue/rationale
- [x] **5.4** Test: reflection returns research_complete after 3+ distinct-domain sources
  - `TestReflectionEnforcement` (9 tests): prompt includes domain count, hard-stop rules, distinct-domain complete
- [x] **5.5** Test: reflection never recommends continuing past max_searches
  - `test_max_searches_enforced_regardless_of_reflection` + `test_max_searches_one_no_reflection`
- [x] **5.6** Test: rationale field is always non-empty
  - `test_rationale_always_non_empty_in_reflection_notes`, `test_rationale_from_failed_reflection_is_non_empty`, `test_rationale_from_exception_is_non_empty`
  - 1876 passed, 6 skipped, 0 failures across full research test suite

---

## Phase 6: Iterative Supervisor Architecture

### Sub-Phase 6.1: State Model Extensions

- [x] **6.1.1** Add `SUPERVISION = "supervision"` to `DeepResearchPhase` enum between GATHERING and ANALYSIS
- [x] **6.1.2** Add supervision fields to `DeepResearchState`: `supervision_round`, `max_supervision_rounds`, `supervision_provider`, `supervision_model`
- [x] **6.1.3** Update `start_new_iteration()` to reset `supervision_round = 0`
- [x] **6.1.4** Update `advance_phase()` docstring to include SUPERVISION
- [x] **6.1.5** Add `should_continue_supervision()` method to `DeepResearchState`

### Sub-Phase 6.2: Configuration

- [x] **6.2.1** Add `deep_research_enable_supervision` (bool, default True) to `ResearchConfig`
- [x] **6.2.2** Add `deep_research_max_supervision_rounds` (int, default 3) to `ResearchConfig`
- [x] **6.2.3** Add `deep_research_supervision_min_sources_per_query` (int, default 2) to `ResearchConfig`
- [x] **6.2.4** Add `"supervision"` to `_ROLE_RESOLUTION_CHAIN` (falls back to `"reflection"`)
- [x] **6.2.5** Add `"supervision"` to `get_phase_timeout()` (reuse planning timeout)
- [x] **6.2.6** Wire `max_supervision_rounds` into `DeepResearchConfig` dataclass and state initialization

### Sub-Phase 6.3: Orchestration Updates

- [x] **6.3.1** Add `DeepResearchPhase.SUPERVISION: AgentRole.SUPERVISOR` to `PHASE_TO_AGENT`
- [x] **6.3.2** Add SUPERVISION case to `_build_agent_inputs()`
- [x] **6.3.3** Add SUPERVISION case to `_evaluate_phase_quality()`
- [x] **6.3.4** Add SUPERVISION to `get_reflection_prompt()` and `_build_reflection_llm_prompt()`

### Sub-Phase 6.4: Supervision Phase Mixin

- [x] **6.4.1** Create `supervision.py` with `SupervisionPhaseMixin` class skeleton
- [x] **6.4.2** Implement `_build_per_query_coverage(state)` — per-sub-query coverage data
- [x] **6.4.3** Implement `_build_supervision_system_prompt(state)` — coverage assessment JSON schema
- [x] **6.4.4** Implement `_build_supervision_user_prompt(state, coverage_data)` — research context + coverage table
- [x] **6.4.5** Implement `_parse_supervision_response(content, state)` — JSON parsing with query dedup
- [x] **6.4.6** Implement `_assess_coverage_heuristic(state, min_sources)` — LLM fallback
- [x] **6.4.7** Implement `_execute_supervision_async(state, provider_id, timeout)` — main entry point

### Sub-Phase 6.5: Workflow Execution Integration

- [x] **6.5.1** Add `_execute_supervision_async` TYPE_CHECKING stub to `WorkflowExecutionMixin`
- [x] **6.5.2** Add SUPERVISION block in `while True` loop after GATHERING block
- [x] **6.5.3** Handle supervision-disabled path: `advance_phase()` to skip SUPERVISION entirely

### Sub-Phase 6.6: Core Class Wiring

- [x] **6.6.1** Add `SupervisionPhaseMixin` to `DeepResearchWorkflow` class inheritance in `core.py`
- [x] **6.6.2** Add import and export in `phases/__init__.py`

### Sub-Phase 6.7: Tests (25 tests)

- [x] **6.7.1** State model tests (6 tests)
- [x] **6.7.2** Prompt/coverage tests (3 tests)
- [x] **6.7.3** Response parsing tests (4 tests)
- [x] **6.7.4** Integration tests (4 tests)
- [x] **6.7.5** Full regression: `pytest tests/core/research/ -x` — 1901 passed, 6 skipped, 0 failures

### Sub-Phase 6.8: Checklist Update

- [x] **6.8.1** Update `PLAN-CHECKLIST.md` Phase 6 section with completed items

---

## Sign-off

- [x] All phases reviewed and approved
- [x] Tests pass: `pytest tests/core/research/ -x` — 1901 passed, 6 skipped, 0 failures
- [x] No regressions in existing deep research tests
- [ ] Code review completed
