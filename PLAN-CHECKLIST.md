# PLAN-CHECKLIST: Pre-Merge Hardening — Round 3

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-26

Track completion of each fix. Mark `[x]` when implemented and verified.

---

## Phase 1: Sanitization Consistency — Synthesis, Compression & Researcher Prompts

- [x] **1A.1** Import `sanitize_external_content` in `synthesis.py` (already imported)
- [x] **1A.2** Sanitize `state.original_query` at ~lines 694, 950, 1107, 1116
- [x] **1A.3** Sanitize `state.research_brief` at ~line 697
- [x] **1A.4** Sanitize `topic_label` (from `sq.query`) at ~line 765
- [x] **1A.5** Sanitize `tr.compressed_findings` before interpolation into synthesis prompt (already sanitized at line 768)
- [x] **1A.6** Sanitize `state.raw_notes` entries in degraded-mode prompt (already sanitized at line 719)
- [x] **1A.7** Sanitize source titles/snippets in synthesis source-listing sections (already sanitized at lines 893, 903, 911, 930)
- [x] **1A.8** Sanitize `tr.raw_notes` in the degraded findings path (~line 723) (already sanitized via comprehension at line 719)
- [x] **1B.1** Import `sanitize_external_content` in `compression.py` (already imported)
- [x] **1B.2** Sanitize `f.content`, `c.description`, `c.resolution`, `g.description`, `s.title`, `state.original_query` in `_execute_global_compression_async`
- [x] **1B.3** Sanitize `query_text` and `src.url` in `_build_message_history_prompt`
- [x] **1B.4** Sanitize `query_text`, `refined_q`, `completion_rationale`, and reflection notes in `_build_structured_metadata_prompt`
- [x] **1B.5** Sanitize raw notes content before prompt inclusion (reflection_notes sanitized in 1B.4)
- [x] **1C.1** Sanitize `topic` parameter in `_build_react_user_prompt` (~line 388)
- [x] **1C.2** Sanitize `content` field of tool-result messages in `_build_react_user_prompt` (~lines 390-400)
- [x] **1C.3** Verify and document that `_format_source_block` already sanitizes (lines 67-70)
- [x] **1D.1** Sanitize `compressed_findings_excerpt` in `build_combined_think_delegate_user_prompt` (~lines 197-200)
- [x] **1D.2** Sanitize `findings_summary` in `build_combined_think_delegate_user_prompt` (~lines 197-200)
- [x] **1D.3** Sanitize `compressed_findings_excerpt` and `findings_summary` in `build_delegation_user_prompt` (~lines 340-343)
- [x] **1D.4** Sanitize `findings_summary` in `build_think_prompt` (~line 779)
- [x] **1D.5** Sanitize `critique_text` and `directives_json` in `build_revision_user_prompt` (~lines 680-703)
- [x] **1D.6** Sanitize clarification constraint keys at ~lines 452-454, 553-556
- [x] **1E.1** Expand `_INJECTION_TAG_PATTERN` with `example|result|output|user|role|artifact|search_results|function_declaration|function_response`
- [x] **1E.2** Add zero-width Unicode character stripping (U+200B, U+200C, U+200D, U+FEFF) before regex matching
- [x] **1E.3** Add HTML entity decoding step (`&lt;` → `<`) before regex matching
- [x] **1F.1** Add test: injection payload in synthesis `original_query` is stripped
- [x] **1F.2** Add test: injection payload in compression `f.content` is stripped
- [x] **1F.3** Add test: injection payload in researcher `topic` is stripped
- [x] **1F.4** Add test: HTML entity-encoded `&lt;system&gt;` is decoded and stripped
- [x] **1F.5** Add test: zero-width character obfuscation in `<system>` tag is handled

**Verification:** `pytest tests/core/research/ -x -q --tb=short -k "sanitize"` ✅ All pass

---

## Phase 2: State Management Bugs

- [x] **2A.1** In `_post_round_bookkeeping`, set `"should_continue_gathering"` to `round_new_sources > 0`
- [x] **2A.2** Add test verifying history entry reflects actual termination decision
- [x] **2B.1** Add optional `suffix` parameter to `store_coverage_snapshot`
- [x] **2B.2** Store pre-directive snapshot with suffix `"pre"` (~line 249)
- [x] **2B.3** Store post-directive snapshot with suffix `"post"` in `_post_round_bookkeeping` (~line 620)
- [x] **2B.4** Update `compute_coverage_delta` to compare `"{round}_pre"` against `"{prev_round}_post"`
- [x] **2B.5** Add test verifying delta accurately reflects inter-round progress
- [x] **2C.1** Add docstring to cancellation rollback block explaining limitations
- [x] **2C.2** Set `state.metadata["rollback_note"] = "partial_iteration_data_retained"` on rollback
- [x] **2C.3** Log warning when rollback occurs
- [x] **2C.4** Add test verifying `rollback_note` metadata is set on cancellation
- [x] **2D.1** Refactor `_should_exit_heuristic` to return `(should_exit: bool, decision_data: dict)`
- [x] **2D.2** Move audit write, history append, and round increment to calling code (~lines 252-253)
- [x] **2D.3** Verify existing heuristic tests still pass

**Verification:** `pytest tests/core/research/workflows/deep_research/test_supervision.py -x -q --tb=short` ✅ All 179 pass

---

## Phase 3: Validation & Config Hardening

- [x] **3A.1** Add `model_validator` to `DelegationResponse`: if `not research_complete` and `not directives`, set `research_complete = True`
- [x] **3A.2** Add `model_validator` to `ReflectionDecision`: if `research_complete`, force `continue_searching = False`
- [x] **3A.3** Add tests for both validators
- [x] **3B.1** Add `deep_research_enable_planning_critique: bool = True` to `ResearchConfig`
- [x] **3B.2** Add corresponding entry in `from_toml_dict()`
- [x] **3B.3** Replace `getattr` in `planning.py` (~line 211) with direct access
- [x] **3C.1** Compute `unknown_keys` after processing all known/deprecated fields in `from_toml_dict()`
- [x] **3C.2** Log warning for each unknown key
- [x] **3C.3** Add test with typo'd key verifying warning fires
- [x] **3D.1** Wrap priority extraction in `_parse_planning_response` (~line 475) in try/except `(ValueError, TypeError)`
- [x] **3D.2** Add test with `"priority": "high"` input
- [x] **3E.1** Add `validate_extract_url()` function to `_helpers.py`
- [x] **3E.2** Block non-HTTP(S) schemes, private IPs, loopback, cloud metadata, link-local
- [x] **3E.3** Call `validate_extract_url` in `_handle_extract_tool`
- [x] **3E.4** Call `validate_extract_url` in `ReflectionDecision._coerce_urls`
- [x] **3E.5** Add tests for each blocked URL pattern

**Verification:** `pytest tests/core/research/ tests/unit/ -x -q --tb=short` ✅ 5801 passed, 6 skipped

---

## Phase 4: Test Coverage Gaps

- [x] **4A.1** Create `tests/core/research/workflows/deep_research/test_workflow_execution.py`
- [x] **4A.2** Test: BRIEF → SUPERVISION phase sequence for new workflows (skip GATHERING)
- [x] **4A.3** Test: cancellation between phases triggers correct rollback
- [x] **4A.4** Test: error in one phase doesn't skip cleanup/state saving
- [x] **4A.5** Test: legacy resume from GATHERING enters gathering, then advances to SYNTHESIS (note: double advance_phase() skips SUPERVISION — GATHERING already does equivalent work)
- [x] **4B.1** Add supervision wall-clock timeout test (set timeout to 0.0s, verify early exit)
- [x] **4B.2** Verify audit event and metadata record wall-clock exit reason
- [x] **4C.1** Add test: all directives in a batch fail → graceful degradation
- [x] **4C.2** Verify: supervision loop exits, state is saved, no crash
- [x] **4D.1** Create canonical `make_test_state(**overrides)` in `conftest.py`
- [x] **4D.2** Migrate `_make_state()` in `test_supervision.py` to use shared helper
- [x] **4D.3** Migrate `_make_state()` in `test_inline_compression.py`
- [x] **4D.4** N/A — `test_topic_compression.py` does not exist (plan referenced wrong filename)
- [x] **4D.5** Migrate `_make_state()` in `test_phase_token_recovery.py`
- [x] **4D.6** Migrate `_make_state_with_existing_sources()` in `test_novelty_tagging.py`
- [x] **4D.7** N/A — `test_structured_outputs.py` does not use `_make_state()`

**Verification:** `pytest tests/core/research/ -x -q --tb=short` ✅ 2571 passed, 6 skipped

---

## Phase 5: Performance & Cleanup

- [ ] **5A.1** Replace `list.pop(0)` loop in `supervision.py` `_trim_raw_notes` with slice assignment
- [ ] **5A.2** Replace `list.pop(0)` loop in `topic_research.py` `_truncate_researcher_history` with slice assignment
- [ ] **5B.1** Build `sub_query_id → list[source]` dict in `build_per_query_coverage`
- [ ] **5B.2** Same optimization in `assess_coverage_heuristic`
- [ ] **5C.1** Move `_FALLBACK_CONTEXT_WINDOW` to `_lifecycle.py` as canonical source
- [ ] **5C.2** Import it in `synthesis.py` instead of redefining
- [ ] **5D.1** Remove inner `import re` in `_parse_combined_response`
- [ ] **5D.2** Remove inner `import re` in `_extract_gap_analysis_section`
- [ ] **5E.1** Replace `find("## SUPERVISOR BRIEF")` with `rfind` in `compression.py`
- [ ] **5E.2** Add check that `compressed` is non-empty after split

**Verification:** `pytest tests/core/research/ -x -q --tb=short`

---

## Sign-off

| Phase | Status | Verified By |
|-------|--------|-------------|
| Phase 1: Sanitization | ✅ Complete | 2026-02-26 |
| Phase 2: State Bugs | ✅ Complete | 2026-02-26 |
| Phase 3: Validation | ✅ Complete | 2026-02-26 |
| Phase 4: Tests | ✅ Complete | 2026-02-26 |
| Phase 5: Performance | ⬜ Not started | |
