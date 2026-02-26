# PLAN-CHECKLIST: Review Findings Fix Plan

## Phase 1: Critical & High-Priority Fixes

- [ ] **1a.** Fix `_evaluate_research` deadlock risk (CRITICAL)
  - [ ] Refactor async dispatch to use `_run_sync` pattern (ThreadPoolExecutor)
  - [ ] `action_handlers.py:573-585` — replace `run_coroutine_threadsafe` + `future.result()`
  - [ ] Add test: call `_evaluate_research` from async context without deadlock

- [ ] **1b.** Restore `CancelledError` propagation (HIGH)
  - [ ] `workflow_execution.py:379-485` — re-raise `CancelledError` after state cleanup
  - [ ] Verify `background_tasks.py` guard handles re-raised exception
  - [ ] Add test: `Task.cancel()` propagates `CancelledError` to caller

- [ ] **1c.** Fix legacy resume crash for PLANNING phase (HIGH)
  - [ ] `workflow_execution.py` — add explicit PLANNING→skip-to-SUPERVISION handling
  - [ ] `phases/__init__.py` — clean up deprecated export
  - [ ] Verify `DeepResearchPhase.PLANNING` enum value still exists for deserialization
  - [ ] Add test: saved state at PLANNING phase resumes without `AttributeError`

- [ ] **1d.** Harden SSRF validation (HIGH)
  - [ ] `_helpers.py:809-857` — add `resolve_dns` parameter to `validate_extract_url()`
  - [ ] Document TOCTOU gap in docstring
  - [ ] `deep_research.py:232-248` — call with `resolve_dns=True` in URL coercion validator
  - [ ] Add test: DNS rebinding detection

## Phase 2: Dead Code Cleanup

- [ ] **2a.** Remove `AnalysisPhaseMixin` from MRO
  - [ ] `core.py:115` — remove from base classes
  - [ ] `phases/__init__.py` — mark export as deprecated

- [ ] **2b.** Remove `RefinementPhaseMixin` from MRO
  - [ ] `core.py:117` — remove from base classes
  - [ ] `phases/__init__.py` — mark export as deprecated

- [ ] **2c.** Delete `supervision_legacy.py` (627 LOC)
  - [ ] Remove file
  - [ ] Remove any imports/references

- [ ] **2d.** Clean up `PlanningPhaseMixin` export
  - [ ] `phases/__init__.py:12,21` — remove from `__all__`

## Phase 3: Type Safety & Config Hardening

- [ ] **3a.** Wire `DeepResearchWorkflowProtocol` into phase mixins
  - [ ] Extend `_protocols.py` protocol with missing method stubs
  - [ ] Update supervision.py mixin to use protocol
  - [ ] Update topic_research.py mixin to use protocol
  - [ ] Update synthesis.py mixin to use protocol
  - [ ] Update compression.py mixin to use protocol
  - [ ] Update brief.py mixin to use protocol
  - [ ] Update gathering.py mixin to use protocol
  - [ ] Update clarification.py mixin to use protocol
  - [ ] Remove duplicated `config: Any` / `memory: Any` stubs from each

- [ ] **3b.** Replace `getattr(self.config, ...)` with direct access
  - [ ] `action_handlers.py` — 4 occurrences
  - [ ] `phases/supervision.py` — occurrences
  - [ ] `phases/topic_research.py` — occurrences
  - [ ] `phases/gathering.py` — occurrences
  - [ ] `audit.py` — occurrences
  - [ ] `persistence.py` — occurrences

## Phase 4: Module-Level Side Effects & Bounded Growth

- [ ] **4a.** Lazy-load `MODEL_TOKEN_LIMITS`
  - [ ] `_lifecycle.py:96-130` — replace module-level load with `@lru_cache`
  - [ ] Update all call sites to use `get_model_token_limits()`

- [ ] **4b.** Promote `_FALLBACK_CONTEXT_WINDOW` to public
  - [ ] `_lifecycle.py:511` — rename to `FALLBACK_CONTEXT_WINDOW`
  - [ ] `synthesis.py:34` — update import
  - [ ] Update internal usages in `_lifecycle.py`

- [ ] **4c.** Post-round supervision message truncation
  - [ ] `supervision.py` — add truncation at end of `_post_round_bookkeeping()` or `_execute_and_merge_directives()`

## Phase 5: Test Improvements

- [ ] **5a.** Add topic research error path tests
  - [ ] Test unknown tool name dispatch
  - [ ] Test budget exhaustion forced termination
  - [ ] Test `extract_content` failure handling

- [ ] **5b.** Add supervision to cross-phase integration test
  - [ ] Add `SupervisionPhaseMixin` to `StubWorkflow`
  - [ ] Add full-path test case: CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS

- [ ] **5c.** Fix `test_supervision_skipped_when_disabled`
  - [ ] Rewrite to call actual production code instead of reimplementing logic inline

- [ ] **5d.** Build evaluation mock responses from `DIMENSIONS` keys
  - [ ] `test_evaluator.py` — dynamically construct dimension names

## Phase 6: Low-Priority Cleanup

- [ ] **6a.** Extract `build_sanitized_context(state)` helper
  - [ ] Add to `_helpers.py`
  - [ ] Update `brief.py`, `planning.py`, `supervision_prompts.py`

- [ ] **6b.** Fix `extract_json()` backslash handling
  - [ ] `_helpers.py:50-77` — only skip on `\` when inside a string
  - [ ] Add test for backslash outside JSON string

- [ ] **6c.** Add JSON/fallback token limits sync test
  - [ ] Verify `model_token_limits.json` keys match `_FALLBACK_MODEL_TOKEN_LIMITS`

- [ ] **6d.** Consolidate `_make_state()` test helpers
  - [ ] Add preset factories to `conftest.py`
  - [ ] Migrate test files incrementally

---

**Total items**: 42 checklist items across 6 phases
**Blocking items**: Phase 1 (4 items) should be completed before merge
**Non-blocking**: Phases 2-6 can be done post-merge or in follow-up PRs
