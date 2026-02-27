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
- [x] **2.1A** Extract sub-methods from `_first_round_decompose_critique_revise()` in `supervision.py` — 84 lines (extracted `_run_first_round_generate`, `_run_first_round_critique`, `_run_first_round_revise`)
- [x] **2.1B** Extract sub-methods from `_execute_supervision_delegation_async()` in `supervision.py` — 123 lines (extracted `_record_supervision_exit`, `_build_delegation_result`, `_prepare_round_coverage`)
- [x] **2.1C** Extract sub-methods from `_execute_topic_research_async()` in `topic_research.py` — 140 lines (extracted `_resolve_topic_research_config`, `_execute_researcher_llm_call`, `_inject_reflection_prompt`, `_parse_with_retry_async`, `_check_reflection_needed`, `_dispatch_tool_calls`, `_build_raw_notes_from_history`, `_apply_inline_compression_async`, `_finalize_topic_result`)
- [x] **2.1D** Extract sub-methods from `_execute_synthesis_async()` in `synthesis.py` — 81 lines (extracted `_handle_empty_findings`, `_prepare_synthesis_budget_and_prompts`, `_execute_synthesis_llm_with_retry`, `_apply_findings_truncation`, `_finalize_synthesis_report`)
- [x] **2.1E** Verify no method exceeds 150 lines post-extraction — all 4 under 150
- [x] **2.1F** Run full test suite — 6777 passed, 0 failed

### 2.2 Split `_helpers.py`
- [x] **2.2A** Create `_json_parsing.py` with `extract_json()`
- [x] **2.2B** Create `_token_budget.py` with truncation/fidelity functions
- [x] **2.2C** Create `_model_resolution.py` with model/provider resolution + dataclasses
- [x] **2.2D** Create `_content_dedup.py` with similarity/novelty functions
- [x] **2.2E** Create `_injection_protection.py` with sanitization + SSRF validation
- [x] **2.2F** Update `_helpers.py` to re-export all symbols for backward compatibility (incl. `_extract_domain`, `_split_prompt_sections`)
- [x] **2.2G** Update direct imports across 16 production files to use new modules
- [x] **2.2H** Verify import resolution: `python -c "from foundry_mcp.core.research.workflows.deep_research._helpers import *"`
- [x] **2.2I** Run full test suite — 6777 passed, 0 failed

### 2.3 Config bounds validation
- [x] **2.3A** Add `_MAX_*` ClassVar constants for each unbounded field
- [x] **2.3B** Add `_validate_deep_research_bounds()` method with warn+clamp pattern
- [x] **2.3C** Wire into existing validation chain (`__post_init__`)
- [x] **2.3D** Add 16 unit tests for bounds clamping and error on invalid values
- [x] **2.3E** Run full test suite — 6777 passed, 0 failed

### 2.4 Coverage heuristic
- [x] **2.4A** Replace `mean()` with `min()` in source adequacy calculation in `assess_coverage_heuristic()`
- [x] **2.4B** Add unit test for lopsided coverage scenario (`test_lopsided_coverage_detected_by_min`)
- [x] **2.4C** Run full test suite — 6777 passed, 0 failed

### 2.5 Token limits validation
- [x] **2.5A** Add `>= 1000` validation in token limits loader (`_load_model_token_limits`)
- [x] **2.5B** Log warning and skip invalid entries
- [x] **2.5C** Add 2 unit tests for malformed token limit values
- [x] **2.5D** Run full test suite — 6777 passed, 0 failed

## Phase 3: NICE-TO-HAVE

### 3.1 Prompt deduplication
- [x] **3.1A** Extract `_build_delegation_core_prompt()` shared helper in `supervision_prompts.py`
- [x] **3.1B** Refactor both `build_*_delegation_system_prompt()` to use shared helper
- [x] **3.1C** Run full test suite — 6777 passed, 0 failed

### 3.2 Centralize round increment
- [x] **3.2A** Create `_advance_supervision_round(state)` static method in `supervision.py`
- [x] **3.2B** Replace all 2 `state.supervision_round += 1` sites with method call (`_record_supervision_exit`, `_post_round_bookkeeping`)
- [x] **3.2C** Run full test suite — 6777 passed, 0 failed

### 3.3 Consolidate `_CHARS_PER_TOKEN`
- [x] **3.3A** Define `CHARS_PER_TOKEN = 4` in `_token_budget.py`
- [x] **3.3B** Update imports in `_lifecycle.py` and `topic_research.py`
- [x] **3.3C** Replace 3 hardcoded `* 4` in `_token_budget.py` with `* CHARS_PER_TOKEN`; kept `_CHARS_PER_TOKEN` alias in `_lifecycle.py` for test backward compat
- [x] **3.3D** Run full test suite — 6777 passed, 0 failed

### 3.4 Fix RuntimeWarning in test
- [x] **3.4A** Fix unawaited coroutine in `_run_phase` — close executor coroutine on cancellation before re-raising
- [x] **3.4B** Verify warning is gone: `pytest tests/.../test_workflow_execution.py -W error::RuntimeWarning -k cancellation_after_brief` — PASSED
- [x] **3.4C** Run full test suite — 6777 passed, 0 failed

---

## Final Validation

- [x] Full test suite green: `pytest tests/ -q` — 6777 passed, 48 skipped
- [x] No new RuntimeWarnings: `pytest tests/ -W error::RuntimeWarning -q` — 6777 passed, 48 skipped, 2 warnings (PytestUnraisableExceptionWarning only)
- [x] No remaining unsanitized trust boundary crossings (grep for direct `state.original_query` / `source.title` / `source.url` interpolation in prompt strings)
