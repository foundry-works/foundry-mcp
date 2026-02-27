# Review Remediation Checklist

Track progress against `PLAN.md`. Mark items `[x]` when complete.

---

## Phase 1: MUST-FIX

- [x] **1.1** Remove `_execute_analysis_async` from expected methods in `test_workflow_inherits_all_phase_methods`
- [x] **1.1a** Verify no other tests reference `_execute_analysis_async`
- [x] **1.1b** Also removed `_execute_refinement_async` (also folded into supervision)
- [x] **1.2A** Sanitize `state.original_query` in `_analysis_prompts.py:103`
- [x] **1.2B** Sanitize `state.research_brief` in `_analysis_prompts.py:106`
- [x] **1.2C** Sanitize `source.title` in `_analysis_prompts.py:141`
- [x] **1.2D** Sanitize `source.title` in `_analysis_prompts.py:215`
- [x] **1.2E** Sanitize `source.url` in `_analysis_prompts.py:217`
- [x] **1.2F** Sanitize `state.original_query` in `refinement.py:338`
- [x] **1.2G** Sanitize `source.url` in `synthesis.py:905`
- [x] **1.2H** Sanitize `source.url` in `synthesis.py:942`
- [x] **1.2I** Sanitize `src.url` in `compression.py:185`
- [x] **1.2J** Add imports for `sanitize_external_content` / `build_sanitized_context` where missing
- [x] **1.2K** Run sanitization tests to verify: `pytest -k "sanitiz" -v` — 141 passed
- [x] **1.2L** Run full test suite: `pytest tests/ -x -q` — 6758 passed, 0 failed

## Phase 2: SHOULD-FIX

### 2.1 Method complexity reduction
- [ ] **2.1A** Extract sub-methods from `_first_round_decompose_critique_revise()` in `supervision.py`
- [ ] **2.1B** Extract sub-methods from `_execute_supervision_delegation_async()` in `supervision.py`
- [ ] **2.1C** Extract sub-methods from `_execute_topic_research_async()` in `topic_research.py`
- [ ] **2.1D** Extract sub-methods from `_execute_synthesis_async()` in `synthesis.py`
- [ ] **2.1E** Verify no method exceeds 150 lines post-extraction
- [ ] **2.1F** Run full test suite — all passing

### 2.2 Split `_helpers.py`
- [ ] **2.2A** Create `_json_parsing.py` with `extract_json()`
- [ ] **2.2B** Create `_token_budget.py` with truncation/fidelity functions
- [ ] **2.2C** Create `_model_resolution.py` with model/provider resolution + dataclasses
- [ ] **2.2D** Create `_content_dedup.py` with similarity/novelty functions
- [ ] **2.2E** Create `_injection_protection.py` with sanitization + SSRF validation
- [ ] **2.2F** Update `_helpers.py` to re-export all symbols for backward compatibility
- [ ] **2.2G** Update direct imports across codebase to use new modules
- [ ] **2.2H** Verify import resolution: `python -c "from foundry_mcp.core.research.workflows.deep_research._helpers import *"`
- [ ] **2.2I** Run full test suite — all passing

### 2.3 Config bounds validation
- [ ] **2.3A** Add `_MAX_*` ClassVar constants for each unbounded field
- [ ] **2.3B** Add `_validate_deep_research_bounds()` method with warn+clamp pattern
- [ ] **2.3C** Wire into existing validation chain
- [ ] **2.3D** Add unit tests for bounds clamping and error on invalid values
- [ ] **2.3E** Run full test suite — all passing

### 2.4 Coverage heuristic
- [ ] **2.4A** Replace `mean()` with `min()` in source adequacy calculation in `assess_coverage_heuristic()`
- [ ] **2.4B** Update/add unit tests for lopsided coverage scenario
- [ ] **2.4C** Run full test suite — all passing

### 2.5 Token limits validation
- [ ] **2.5A** Add `>= 1000` validation in token limits loader
- [ ] **2.5B** Log warning and skip invalid entries
- [ ] **2.5C** Add unit test for malformed token limit values
- [ ] **2.5D** Run full test suite — all passing

## Phase 3: NICE-TO-HAVE

### 3.1 Prompt deduplication
- [ ] **3.1A** Extract `_build_delegation_core_prompt()` shared helper in `supervision_prompts.py`
- [ ] **3.1B** Refactor both `build_*_delegation_system_prompt()` to use shared helper
- [ ] **3.1C** Run full test suite — all passing

### 3.2 Centralize round increment
- [ ] **3.2A** Create `_advance_supervision_round(state)` method in `supervision.py`
- [ ] **3.2B** Replace all 4 `state.supervision_round += 1` sites with method call
- [ ] **3.2C** Run full test suite — all passing

### 3.3 Consolidate `_CHARS_PER_TOKEN`
- [ ] **3.3A** Define `CHARS_PER_TOKEN = 4` in `_token_budget.py` (or `_helpers.py`)
- [ ] **3.3B** Update imports in `_lifecycle.py` and `topic_research.py`
- [ ] **3.3C** Remove duplicate definitions
- [ ] **3.3D** Run full test suite — all passing

### 3.4 Fix RuntimeWarning in test
- [ ] **3.4A** Fix unawaited coroutine in `test_cancellation_after_brief_before_supervision`
- [ ] **3.4B** Verify warning is gone: `pytest tests/.../test_workflow_execution.py -W error::RuntimeWarning -k cancellation_after_brief`
- [ ] **3.4C** Run full test suite — all passing

---

## Final Validation

- [ ] Full test suite green: `pytest tests/ -q`
- [ ] No new RuntimeWarnings: `pytest tests/ -W error::RuntimeWarning -q`
- [ ] No remaining unsanitized trust boundary crossings (grep for direct `state.original_query` / `source.title` / `source.url` interpolation in prompt strings)
