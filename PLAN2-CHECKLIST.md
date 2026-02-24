# PLAN2-CHECKLIST: Phase 6 — Iterative Supervisor Architecture

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23

---

## Sub-Phase 6.1: State Model Extensions

- [x] **6.1.1** Add `SUPERVISION = "supervision"` to `DeepResearchPhase` enum between GATHERING and ANALYSIS
- [x] **6.1.2** Add supervision fields to `DeepResearchState`: `supervision_round`, `max_supervision_rounds`, `supervision_provider`, `supervision_model`
- [x] **6.1.3** Update `start_new_iteration()` to reset `supervision_round = 0`
- [x] **6.1.4** Update `advance_phase()` docstring to include SUPERVISION
- [x] **6.1.5** Add `should_continue_supervision()` method to `DeepResearchState`

---

## Sub-Phase 6.2: Configuration

- [ ] **6.2.1** Add `deep_research_enable_supervision` (bool, default True) to `ResearchConfig`
- [ ] **6.2.2** Add `deep_research_max_supervision_rounds` (int, default 3) to `ResearchConfig`
- [ ] **6.2.3** Add `deep_research_supervision_min_sources_per_query` (int, default 2) to `ResearchConfig`
- [ ] **6.2.4** Add `"supervision"` to `_ROLE_RESOLUTION_CHAIN` (falls back to `"reflection"`)
- [ ] **6.2.5** Add `"supervision"` to `get_phase_timeout()` (reuse planning timeout)
- [ ] **6.2.6** Wire `max_supervision_rounds` into `DeepResearchConfig` dataclass and state initialization

---

## Sub-Phase 6.3: Orchestration Updates

- [ ] **6.3.1** Add `DeepResearchPhase.SUPERVISION: AgentRole.SUPERVISOR` to `PHASE_TO_AGENT`
- [ ] **6.3.2** Add SUPERVISION case to `_build_agent_inputs()`
- [ ] **6.3.3** Add SUPERVISION case to `_evaluate_phase_quality()`
- [ ] **6.3.4** Add SUPERVISION to `get_reflection_prompt()` and `_build_reflection_llm_prompt()`

---

## Sub-Phase 6.4: Supervision Phase Mixin

- [ ] **6.4.1** Create `supervision.py` with `SupervisionPhaseMixin` class skeleton
- [ ] **6.4.2** Implement `_build_per_query_coverage(state)` — per-sub-query coverage data
- [ ] **6.4.3** Implement `_build_supervision_system_prompt(state)` — coverage assessment JSON schema
- [ ] **6.4.4** Implement `_build_supervision_user_prompt(state, coverage_data)` — research context + coverage table
- [ ] **6.4.5** Implement `_parse_supervision_response(content, state)` — JSON parsing with query dedup
- [ ] **6.4.6** Implement `_assess_coverage_heuristic(state, min_sources)` — LLM fallback
- [ ] **6.4.7** Implement `_execute_supervision_async(state, provider_id, timeout)` — main entry point
  - Heuristic early-exit when all queries sufficiently covered
  - LLM call via `execute_llm_call()` with `role="supervision"`
  - Follow-up query dedup and sub-query budget cap
  - History recording in `state.metadata["supervision_history"]`

---

## Sub-Phase 6.5: Workflow Execution Integration

- [ ] **6.5.1** Add `_execute_supervision_async` TYPE_CHECKING stub to `WorkflowExecutionMixin`
- [ ] **6.5.2** Add SUPERVISION block in `while True` loop after GATHERING block
  - `skip_transition=True` for manual transition control
  - Loop-back to GATHERING when `should_continue_gathering` and pending queries within limit
  - `advance_phase()` to ANALYSIS when supervision is satisfied
- [ ] **6.5.3** Handle supervision-disabled path: `advance_phase()` to skip SUPERVISION entirely

---

## Sub-Phase 6.6: Core Class Wiring

- [ ] **6.6.1** Add `SupervisionPhaseMixin` to `DeepResearchWorkflow` class inheritance in `core.py`
- [ ] **6.6.2** Add import and export in `phases/__init__.py`

---

## Sub-Phase 6.7: Tests

- [ ] **6.7.1** State model tests (6 tests)
  - `test_supervision_phase_in_enum`
  - `test_advance_phase_gathering_to_supervision`
  - `test_advance_phase_supervision_to_analysis`
  - `test_start_new_iteration_resets_supervision_round`
  - `test_should_continue_supervision_within_limit`
  - `test_should_continue_supervision_at_limit`
- [ ] **6.7.2** Prompt/coverage tests (3 tests)
  - `test_build_per_query_coverage`
  - `test_supervision_system_prompt_has_json_schema`
  - `test_supervision_user_prompt_includes_coverage`
- [ ] **6.7.3** Response parsing tests (4 tests)
  - `test_parse_valid_supervision_response`
  - `test_parse_with_duplicate_queries_deduped`
  - `test_parse_invalid_json_returns_fallback`
  - `test_heuristic_fallback`
- [ ] **6.7.4** Integration tests (4 tests)
  - `test_supervision_loop_adds_follow_up_queries`
  - `test_supervision_loop_terminates_at_max_rounds`
  - `test_supervision_skipped_when_disabled`
  - `test_supervision_proceeds_when_all_covered`
- [ ] **6.7.5** Full regression: `pytest tests/core/research/ -x` — no regressions

---

## Sub-Phase 6.8: Checklist Update

- [ ] **6.8.1** Update `PLAN-CHECKLIST.md` Phase 6 section with completed items

---

## Sign-off

- [ ] All sub-phases reviewed and approved
- [ ] Tests pass: `pytest tests/core/research/ -x`
- [ ] No regressions in existing deep research tests
- [ ] Supervision loop verified: GATHERING → SUPERVISION → (loop) → ANALYSIS
- [ ] Supervision-disabled path verified: GATHERING → ANALYSIS (unchanged behavior)
