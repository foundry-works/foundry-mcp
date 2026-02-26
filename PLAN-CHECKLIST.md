# PLAN-CHECKLIST: Pre-Merge Fixes from Senior Engineer Review

Track completion of each fix. Mark `[x]` when implemented and verified.

---

## Phase 1 — Architectural: Split `supervision.py`

- [x] **1.1** Extract prompt builders into `supervision_prompts.py`
  - [x] Create `phases/supervision_prompts.py` with module-level functions
  - [x] Move `_build_delegation_system_prompt` / `_build_delegation_user_prompt`
  - [x] Move `_build_combined_think_delegate_system_prompt` / `_build_combined_think_delegate_user_prompt`
  - [x] Move `_build_first_round_think_prompt` / `_build_first_round_delegation_system_prompt` / `_build_first_round_delegation_user_prompt`
  - [x] Move `_build_critique_system_prompt` / `_build_critique_user_prompt`
  - [x] Move `_build_revision_system_prompt` / `_build_revision_user_prompt`
  - [x] Move `_build_think_prompt`
  - [x] Update imports in `supervision.py` to call extracted functions
  - [x] Verify all supervision tests still pass
- [x] **1.2** Extract coverage assessment into `supervision_coverage.py`
  - [x] Create `phases/supervision_coverage.py` with module-level functions
  - [x] Move `_build_per_query_coverage`
  - [x] Move `_store_coverage_snapshot` / `_compute_coverage_delta`
  - [x] Move `_assess_coverage_heuristic`
  - [x] Move `_VERDICT_*` / `_ISSUE_MARKER_RE` patterns and `_critique_has_issues`
  - [x] Update imports in `supervision.py`
  - [x] Verify coverage heuristic tests still pass
- [x] **1.3** Decompose `_execute_supervision_delegation_async`
  - [x] Extract `_should_exit_wall_clock()` helper
  - [x] Extract `_should_exit_heuristic()` helper
  - [x] Extract `_run_think_delegate_step()` helper
  - [x] Extract `_execute_and_merge_directives()` helper
  - [x] Extract `_post_round_bookkeeping()` helper
  - [x] Main loop reduced to ~90 lines calling sub-methods (includes inline early-exit blocks)
  - [x] Verify delegation integration tests still pass
- [x] **1.4** Extract duplicate supervision message rendering
  - [x] Create `_render_supervision_conversation_history()` in `supervision_prompts.py`
  - [x] Replace duplicate code in `_build_delegation_user_prompt` (lines 1599-1633)
  - [x] Replace duplicate code in `_build_combined_think_delegate_user_prompt` (lines 909-943)
- [x] **1.5** Move inline import of `_normalize_title` to module level
  - [x] Move import from inside `_execute_directives_async` (line 1071) to module-level block

**Verification:** `pytest tests/core/research/workflows/deep_research/test_supervision.py -x -q`

---

## Phase 2 — Security: Sanitization Consistency

- [x] **2.1** Sanitize `state.original_query` in supervision prompt builders
  - [x] `_build_delegation_user_prompt`: wrap `state.original_query` in `sanitize_external_content()`
  - [x] `_build_combined_think_delegate_user_prompt`: same
  - [x] `_build_first_round_think_prompt`: same
  - [x] `_build_first_round_delegation_user_prompt`: same
  - [x] `_build_critique_user_prompt`: same
  - [x] `_build_revision_user_prompt`: same
  - [x] `_build_think_prompt`: same
  - [x] Also sanitize `state.research_brief` where interpolated in supervision prompts
  - [x] Also sanitize `state.system_prompt` where interpolated in supervision prompts
- [x] **2.2** Sanitize `state.original_query` in planning prompt builders
  - [x] `_build_brief_refinement_prompt` (line 357): wrap `state.original_query`
  - [x] `_build_brief_refinement_prompt` (lines 361, 365-366): wrap `state.system_prompt` and constraints
  - [x] `_build_planning_user_prompt` (line 386): wrap `state.research_brief`
  - [x] `_build_planning_user_prompt` (line 400): wrap `state.system_prompt`
  - [x] `_build_decomposition_critique_prompt` (line 535): wrap `state.research_brief or state.original_query`
- [x] **2.3** Sanitize second-order injection vectors in supervision
  - [x] Sanitize `entry['query']` in `_build_delegation_user_prompt` coverage section (line 1640)
  - [x] Sanitize `entry['query']` in `_build_combined_think_delegate_user_prompt` (line 950)
  - [x] Sanitize `d.research_topic` in "Previously Executed Directives" (lines 1683, 962)

**Verification:** `pytest tests/core/research/workflows/deep_research/test_sanitize_external_content.py -x -q`

---

## Phase 3 — Test Coverage: Restore Deleted Test Coverage

- [x] **3.1** Resolve digest integration test gap
  - [x] Determine if `_execute_digest_step_async` is still reachable from the new pipeline → DEAD CODE
  - [x] Remove `_analysis_digest.py` (deleted file)
  - [x] Remove `DigestStepMixin` from `AnalysisPhaseMixin` inheritance and imports
  - [x] Remove digest step call from `_execute_analysis_async`
- [x] **3.2** Resolve contradiction detection test gap
  - [x] Determine if `_detect_contradictions()` in `analysis.py` is still called → DEAD CODE (config flag deprecated)
  - [x] Remove `_detect_contradictions()` method from `analysis.py`
  - [x] Remove contradiction detection call block from `_execute_analysis_async`
  - [x] Clean up unused imports (`Contradiction`, `asyncio`, `json`)
- [x] **3.3** Resolve orchestration reflection test gap
  - [x] Determine if `PhaseReflectionDecision`, `async_think_pause` are still called → DEAD CODE
  - [x] Remove `PhaseReflectionDecision` dataclass from `orchestration.py`
  - [x] Remove `async_think_pause()`, `_build_reflection_system_prompt()`, `_build_reflection_llm_prompt()`, `_parse_reflection_response()` from `orchestration.py`
  - [x] Remove `dispatch_to_agent()` and `_build_agent_inputs()` (also dead)
  - [x] Clean up unused imports (`json`, `SourceQuality`)
- [x] **3.4** Add dedicated brief phase tests
  - [x] Create `tests/core/research/workflows/deep_research/test_brief.py`
  - [x] Test: LLM failure → workflow continues with original query (non-fatal)
  - [x] Test: malformed JSON → falls back to plain-text brief
  - [x] Test: `ResearchBriefOutput` parsing with missing optional fields
  - [x] Test: brief generation with clarification constraints included
  - [x] Additional: 9 parse_brief_output unit tests, 4 ResearchBriefOutput model tests, 3 prompt sanitization tests

**Verification:** `pytest tests/core/research/ -x -q --tb=short` → 2498 passed, 6 skipped

---

## Phase 4 — Config & Data Model Bugs

- [ ] **4.1** Fix `per_provider_rate_limits` default mismatch
  - [ ] Change `from_toml_dict()` line 379: `"semantic_scholar": 100` → `"semantic_scholar": 20`
  - [ ] Change `samples/foundry-mcp.toml` line 650: `semantic_scholar = 100` → `semantic_scholar = 20`
  - [ ] Add unit test: `ResearchConfig()` and `ResearchConfig.from_toml_dict({})` produce same `semantic_scholar` limit
- [ ] **4.2** Add `deep_research_mode` validation
  - [ ] Add `_validate_deep_research_mode()` to `__post_init__` chain
  - [ ] Validate against `{"general", "academic", "technical"}`
  - [ ] Add unit test: invalid mode raises `ValueError`
- [ ] **4.3** Add missing Gemini 2.0 model entries
  - [ ] Add `"gemini-2.0-pro": 1048576` to `model_token_limits.json`
  - [ ] Add `"gemini-2.0-flash": 1048576` to `model_token_limits.json`
  - [ ] Add matching entries to `_FALLBACK_MODEL_TOKEN_LIMITS` in `_lifecycle.py`

**Verification:** `pytest tests/unit/test_config_supervision.py tests/core/research/workflows/test_model_routing.py -x -q`

---

## Phase 5 — Cleanup: Code Quality

- [ ] **5.1** Rename `DeepResearchConfig` sub-config to avoid collision
  - [ ] Rename class to `DeepResearchSettings` in `research_sub_configs.py`
  - [ ] Update all import references
  - [ ] Verify no test breakage
- [ ] **5.2** Isolate legacy supervision query-generation model
  - [ ] Create `phases/supervision_legacy.py`
  - [ ] Move `_execute_supervision_query_generation_async` (~253 lines)
  - [ ] Move legacy prompt builders (`_build_supervision_system_prompt`, `_build_supervision_user_prompt`)
  - [ ] Move `_parse_supervision_response`
  - [ ] Import conditionally from `supervision.py`
- [ ] **5.3** Add explicit phase transition table comment
  - [ ] Add phase transition documentation in `workflow_execution.py`
  - [ ] Document legacy GATHERING → SUPERVISION auto-advance
- [ ] **5.4** Add `Protocol` class for mixin interface (nice-to-have)
  - [ ] Create `phases/_protocols.py` with `DeepResearchWorkflowProtocol`
  - [ ] Define `config`, `memory`, `_write_audit_event`, `_check_cancellation`, `_execute_provider_async`
  - [ ] Update mixin TYPE_CHECKING stubs to reference protocol

**Verification:** `pytest tests/ -x -q --tb=short` (full suite)

---

## Summary

| Phase | Items | Priority | Est. LOC |
|-------|-------|----------|----------|
| 1. Architectural split | 5 items | Must-fix | ~200 (moves, not new code) |
| 2. Sanitization consistency | 3 items | Should-fix | ~60 |
| 3. Test coverage gaps | 4 items | Should-fix | ~400-800 (depending on dead code removal vs new tests) |
| 4. Config bugs | 3 items | Should-fix | ~30 |
| 5. Cleanup | 4 items | Nice-to-have | ~150 |

**Total estimated new/changed LOC:** ~840-1240

**Execution order:** Phase 1 first (unblocks future plans). Phases 2-4 can proceed in parallel after Phase 1. Phase 5 last.
