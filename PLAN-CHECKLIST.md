# PLAN-CHECKLIST: Review Findings Fix Plan

## Phase 1: Critical & High-Priority Fixes

- [x] **1a.** Fix `_evaluate_research` deadlock risk (CRITICAL)
  - [x] Refactor async dispatch to use `_run_sync` pattern (ThreadPoolExecutor)
  - [x] `action_handlers.py:573-585` — replace `run_coroutine_threadsafe` + `future.result()`
  - [x] Add test: call `_evaluate_research` from async context without deadlock

- [x] **1b.** Restore `CancelledError` propagation (HIGH)
  - [x] `workflow_execution.py:379-485` — re-raise `CancelledError` after state cleanup
  - [x] Verify `background_tasks.py` guard handles re-raised exception
  - [x] Add test: `Task.cancel()` propagates `CancelledError` to caller

- [x] **1c.** Fix legacy resume crash for PLANNING phase (HIGH)
  - [x] `workflow_execution.py` — add explicit PLANNING→skip-to-SUPERVISION handling
  - [x] `phases/__init__.py` — clean up deprecated export
  - [x] Verify `DeepResearchPhase.PLANNING` enum value still exists for deserialization
  - [x] Add test: saved state at PLANNING phase resumes without `AttributeError`

- [x] **1d.** Harden SSRF validation (HIGH)
  - [x] `_helpers.py:809-857` — add `resolve_dns` parameter to `validate_extract_url()`
  - [x] Document TOCTOU gap in docstring
  - [x] `deep_research.py:232-248` — call with `resolve_dns=True` in URL coercion validator
  - [x] Add test: DNS rebinding detection

## Phase 2: Dead Code Cleanup

- [x] **2a.** Remove `AnalysisPhaseMixin` from MRO
  - [x] `core.py:115` — remove from base classes
  - [x] `phases/__init__.py` — mark export as deprecated

- [x] **2b.** Remove `RefinementPhaseMixin` from MRO
  - [x] `core.py:117` — remove from base classes
  - [x] `phases/__init__.py` — mark export as deprecated

- [x] **2c.** Delete `supervision_legacy.py` (627 LOC)
  - [x] Remove file
  - [x] Remove any imports/references

- [x] **2d.** Clean up `PlanningPhaseMixin` export
  - [x] `phases/__init__.py:12,21` — remove from `__all__`

## Phase 3: Type Safety & Config Hardening

- [x] **3a.** Wire `DeepResearchWorkflowProtocol` into phase mixins
  - [x] Extend `_protocols.py` protocol with missing method stubs
  - [x] Update supervision.py mixin to use protocol
  - [x] Update topic_research.py mixin to use protocol
  - [x] Update synthesis.py mixin to use protocol
  - [x] Update compression.py mixin to use protocol
  - [x] Update brief.py mixin to use protocol
  - [x] Update gathering.py mixin to use protocol
  - [x] Update clarification.py mixin to use protocol
  - [x] Remove duplicated `config: Any` / `memory: Any` stubs from each

- [x] **3b.** Replace `getattr(self.config, ...)` with direct access
  - [x] `action_handlers.py` — 4 occurrences
  - [x] `phases/supervision.py` — 1 occurrence
  - [x] `phases/topic_research.py` — 7 occurrences
  - [x] `phases/gathering.py` — 4 occurrences
  - [x] `audit.py` — 1 occurrence
  - [x] `persistence.py` — 1 occurrence

## Phase 4: Module-Level Side Effects & Bounded Growth

- [x] **4a.** Lazy-load `MODEL_TOKEN_LIMITS`
  - [x] `_lifecycle.py:96-130` — replace module-level load with `@lru_cache`
  - [x] Update all call sites to use `get_model_token_limits()`

- [x] **4b.** Promote `_FALLBACK_CONTEXT_WINDOW` to public
  - [x] `_lifecycle.py:511` — rename to `FALLBACK_CONTEXT_WINDOW`
  - [x] `synthesis.py:34` — update import
  - [x] Update internal usages in `_lifecycle.py`

- [x] **4c.** Post-round supervision message truncation
  - [x] `supervision.py` — add truncation at end of `_post_round_bookkeeping()`

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
