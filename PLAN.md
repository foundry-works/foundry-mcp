# PLAN: Pre-Merge Hardening — Round 3

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-26
**Status:** Draft

---

## Context

Third-round senior engineering review of the full branch (100+ commits, +40k/-7.7k lines across 108 files) after the Round 1–2 hardening passes. The previous rounds fixed architectural splitting, prompt sanitization in supervision prompts, config deprecation warnings, and state growth caps. This round addresses remaining gaps: inconsistent sanitization in synthesis/compression/researcher prompts, state management bugs, validation gaps, test coverage holes, and performance issues.

---

## Phase 1: Sanitization Consistency — Synthesis, Compression & Researcher Prompts

**Effort:** Medium | **Impact:** Critical
**Goal:** Apply `sanitize_external_content()` consistently in the three prompt-building code paths that were missed by the Round 2 sanitization pass. Currently only `supervision_prompts.py` is consistently sanitized.

### 1A. Sanitize synthesis prompt inputs

**Problem:** `synthesis.py` interpolates `state.original_query`, `state.research_brief`, `topic_label` (from `sq.query`), source titles, and raw notes directly into LLM prompts without sanitization. The supervision prompts correctly sanitize these same fields, creating a bypass surface.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

**Changes:**
1. Import `sanitize_external_content` from `_helpers`
2. Sanitize `state.original_query` at lines ~694, ~950, ~1107, ~1116
3. Sanitize `state.research_brief` at line ~697
4. Sanitize `topic_label` (from `sq.query`) at line ~765
5. Sanitize `tr.compressed_findings` before interpolation into synthesis prompt
6. Sanitize `state.raw_notes` entries before joining into degraded-mode prompt
7. Sanitize source titles and snippets in source-listing sections
8. Sanitize `tr.raw_notes` at line ~723 in the degraded findings path

### 1B. Sanitize compression prompt inputs

**Problem:** `compression.py` interpolates finding content, contradiction descriptions, gap descriptions, source titles, query text, reflection notes, and completion rationale without sanitization. Global compression at lines ~920-977 is the largest gap.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Changes:**
1. Import `sanitize_external_content` from `_helpers`
2. In `_execute_global_compression_async` (~lines 920-977): sanitize `f.content`, `c.description`, `c.resolution`, `g.description`, `s.title`, `state.original_query`
3. In `_build_message_history_prompt` (~lines 96-102): sanitize `query_text`, `src.url`
4. In `_build_structured_metadata_prompt` (~lines 144-203): sanitize `query_text`, `refined_q`, `completion_rationale`, and reflection notes
5. Sanitize raw notes content before prompt inclusion

### 1C. Sanitize researcher prompt inputs

**Problem:** `topic_research.py` — `_build_react_user_prompt` embeds `sub_query.query` (the topic) directly into XML tags without sanitization. Message history content from tool results is rebuilt into the prompt without re-sanitization.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

**Changes:**
1. In `_build_react_user_prompt` (~line 388): sanitize `topic` parameter
2. In `_build_react_user_prompt` (~lines 390-400): sanitize `content` field of tool-result messages when rebuilding history into the user prompt
3. Verify `_format_source_block` already sanitizes (it does — lines 67-70) and document the contract

### 1D. Sanitize supervision prompt coverage data

**Problem:** `supervision_prompts.py` — `compressed_findings_excerpt` and `findings_summary` from coverage data, and `critique_text`/`directives_json` in the revision prompt, are interpolated without sanitization. Also, clarification constraint keys are unsanitized.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_prompts.py`

**Changes:**
1. Sanitize `entry['compressed_findings_excerpt']` and `entry['findings_summary']` at ~lines 197-200, 340-343
2. Sanitize `findings_summary` at ~line 779 in `build_think_prompt`
3. Sanitize `critique_text` and `directives_json` in `build_revision_user_prompt` (~lines 680-703)
4. Sanitize clarification constraint keys at ~lines 452-454, 553-556

### 1E. Expand sanitizer tag coverage

**Problem:** `_helpers.py` — The injection tag blocklist misses `<example>`, `<result>`, `<output>`, `<user>`, `<role>`, `<artifact>`, `<search_results>`, and model-specific tags. No handling of HTML entity-encoded or zero-width character obfuscation.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`

**Changes:**
1. Add `example|result|output|user|role|artifact|search_results|function_declaration|function_response` to `_INJECTION_TAG_PATTERN`
2. Add a Unicode normalization step before regex matching: strip zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
3. Add HTML entity decoding step (`&lt;` → `<`) before regex matching

### 1F. Add cross-phase sanitization tests

**Files:**
- `tests/core/research/workflows/deep_research/test_sanitize_external_content.py` (extend existing)

**Changes:**
1. Add tests: injection payload in synthesis prompt inputs is stripped
2. Add tests: injection payload in compression prompt inputs is stripped
3. Add tests: injection payload in researcher topic is stripped
4. Add tests: HTML entity-encoded injection is decoded and stripped
5. Add tests: zero-width character obfuscation is handled

---

## Phase 2: State Management Bugs

**Effort:** Low | **Impact:** High
**Goal:** Fix three state management bugs that corrupt audit data or mislead the supervisor LLM.

### 2A. Fix `should_continue_gathering` always `True` in supervision history

**Problem:** `supervision.py` line ~642 unconditionally records `"should_continue_gathering": True` in the supervision history entry, even when the round terminates with no new sources. This corrupts audit data.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. In `_post_round_bookkeeping`, set `"should_continue_gathering"` based on the actual decision: `round_new_sources > 0`
2. Add a test verifying the history entry reflects the actual decision

### 2B. Fix coverage snapshot overwrite

**Problem:** `supervision.py` — Pre-directive and post-directive coverage snapshots use the same key (`state.supervision_round`). The pre-directive snapshot is silently overwritten. The next round's `compute_coverage_delta` compares against the post-directive snapshot, potentially showing zero progress.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py`

**Changes:**
1. In `store_coverage_snapshot`, accept an optional `suffix` parameter (e.g., `"pre"`, `"post"`)
2. At ~line 249 (main loop), store with suffix `"pre"`
3. In `_post_round_bookkeeping` (~line 620), store with suffix `"post"`
4. In `compute_coverage_delta`, compare `"{round}_pre"` of current round against `"{prev_round}_post"` of previous round
5. Add a test verifying the delta accurately reflects inter-round progress

### 2C. Document cancellation rollback limitations

**Problem:** `workflow_execution.py` lines ~399-421 claims to "rollback" on cancellation but only resets `state.iteration` and `state.phase`. Accumulated sources, findings, and directives from the partial iteration remain in state.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

**Changes:**
1. Add clear docstring/comments explaining that "rollback" only resets the iteration counter and phase marker
2. Add `state.metadata["rollback_note"] = "partial_iteration_data_retained"` so resume logic can detect this condition
3. Log a warning when rollback occurs, explaining the limitation
4. Add a brief test verifying the metadata flag is set on cancellation rollback

### 2D. Separate side effects from `_should_exit_heuristic`

**Problem:** `supervision.py` lines ~391-427 — `_should_exit_heuristic` is named as a predicate but increments `supervision_round`, writes audit events, and appends to supervision history. A predicate should not mutate state.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Have `_should_exit_heuristic` return only a boolean (and the decision data as a second return value)
2. Move the state mutations (audit write, history append, round increment) into the calling code at ~lines 252-253
3. Pattern should match the inline blocks for `research_complete` (~lines 263-279) and `no_directives` (~lines 281-297)

---

## Phase 3: Validation & Config Hardening

**Effort:** Low | **Impact:** Medium
**Goal:** Fix validation gaps in data models and config that allow semantically invalid states or silent misconfiguration.

### 3A. Add model validators for structured output schemas

**Problem:** `DelegationResponse` allows `research_complete=False` with empty `directives`. `ReflectionDecision` allows both `continue_searching=True` and `research_complete=True` simultaneously.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`

**Changes:**
1. Add `model_validator` to `DelegationResponse`: if `not research_complete` and `not directives`, set `research_complete = True` with a logged warning
2. Add `model_validator` to `ReflectionDecision`: if `research_complete is True`, force `continue_searching = False`
3. Add tests for both validators

### 3B. Declare ghost config field `deep_research_enable_planning_critique`

**Problem:** `planning.py` line ~211 reads `self.config.deep_research_enable_planning_critique` via `getattr` with default `True`. This field does not exist on `ResearchConfig`, so users cannot configure it via TOML.

**Files:**
- `src/foundry_mcp/config/research.py`

**Changes:**
1. Add `deep_research_enable_planning_critique: bool = True` to `ResearchConfig`
2. Add corresponding entry in `from_toml_dict()`
3. Remove `getattr` wrapper in `planning.py` — use direct access

### 3C. Warn on unknown TOML keys

**Problem:** `from_toml_dict()` silently ignores typos in TOML config (e.g., `deep_research_max_interations` → uses default with no warning). With 80+ fields, typos are likely.

**Files:**
- `src/foundry_mcp/config/research.py`

**Changes:**
1. After processing all known and deprecated fields, compute `unknown_keys = set(data.keys()) - known_keys - deprecated_keys`
2. Log a warning for each unknown key: `"Unknown config key '{key}' in [research] section — check for typos"`
3. Add a test with a typo'd key verifying the warning fires

### 3D. Fix planning priority parse crash

**Problem:** `planning.py` line ~475 — `int(sq.get("priority", i + 1))` raises `ValueError` if the LLM returns a non-numeric priority (e.g., `"high"`).

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py`

**Changes:**
1. Wrap in try/except: `try: priority = min(max(int(sq.get("priority", i + 1)), 1), 10) except (ValueError, TypeError): priority = i + 1`
2. Add test with `"priority": "high"` input

### 3E. Add SSRF protection for extract URLs

**Problem:** `topic_research.py` — URLs from the researcher LLM are passed to the Tavily Extract API with only `startswith("http")` validation. No blocklist for private IPs, cloud metadata, or non-HTTP schemes.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py` (new function)

**Changes:**
1. Add `validate_extract_url(url: str) -> bool` to `_helpers.py` that rejects:
   - Non-HTTP(S) schemes
   - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
   - Loopback (127.x, localhost, ::1)
   - Cloud metadata (169.254.169.254)
   - Link-local (169.254.x)
2. Call `validate_extract_url` in `_handle_extract_tool` and in `ReflectionDecision._coerce_urls`
3. Add tests for each blocked pattern

---

## Phase 4: Test Coverage Gaps

**Effort:** Medium | **Impact:** Medium
**Goal:** Add tests for the most critical uncovered code paths identified in the review.

### 4A. Add `workflow_execution.py` integration test

**Problem:** The 540-line orchestration loop has almost no direct test coverage. Existing tests reimplement its logic inline rather than calling the actual code.

**Files:**
- `tests/core/research/workflows/deep_research/test_workflow_execution.py` (new file)

**Changes:**
1. Create test class that instantiates the real `WorkflowExecutionMixin` with mocked phase methods
2. Test: BRIEF → SUPERVISION (skip GATHERING) phase sequence for new workflows
3. Test: cancellation between phases triggers correct rollback
4. Test: error in one phase doesn't skip cleanup/state saving
5. Test: legacy resume from GATHERING enters gathering, then advances to SUPERVISION

### 4B. Add supervision wall-clock timeout test

**Problem:** `deep_research_supervision_wall_clock_timeout` is configured in stubs but never actually tested.

**Files:**
- `tests/core/research/workflows/deep_research/test_supervision.py`

**Changes:**
1. Add test with `wall_clock_timeout = 0.1` (very short) that verifies the supervision loop exits early
2. Verify the audit event and metadata record the wall-clock exit reason

### 4C. Add "all directives fail" edge case test

**Problem:** Tests cover individual directive failure and cancellation, but not the case where every directive in a batch fails.

**Files:**
- `tests/core/research/workflows/deep_research/test_supervision.py`

**Changes:**
1. Add test with all parallel directive executions raising exceptions
2. Verify graceful degradation: supervision loop exits, state is saved, no crash

### 4D. Consolidate `_make_state()` test helpers

**Problem:** `_make_state()` is independently defined in 6+ test files with slightly different signatures. Changes to `DeepResearchState` require updating all copies.

**Files:**
- `tests/core/research/workflows/deep_research/conftest.py` (new or existing)
- All test files referencing `_make_state()`

**Changes:**
1. Create canonical `make_test_state(**overrides)` in `conftest.py`
2. Migrate callers in: `test_supervision.py`, `test_inline_compression.py`, `test_topic_compression.py`, `test_phase_token_recovery.py`, `test_novelty_tagging.py`, `test_structured_outputs.py`
3. Keep file-specific specializations as thin wrappers calling the canonical helper

---

## Phase 5: Performance & Cleanup

**Effort:** Low | **Impact:** Low
**Goal:** Fix O(n²) patterns and code quality issues identified in the review.

### 5A. Replace `list.pop(0)` loops with slice assignment

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (~lines 569-599)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (~lines 468-471)

**Changes:**
1. Replace `while ... pop(0)` loops with: compute drop count, then `items = items[drop_count:]`
2. Verify with existing tests

### 5B. Optimize O(n×m) source lookups in coverage assessment

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py` (~lines 84-89)

**Changes:**
1. Build `sub_query_id → list[source]` lookup dict at top of `build_per_query_coverage`
2. Index into it instead of filtering per query
3. Same optimization in `assess_coverage_heuristic` (~line 328)

### 5C. Deduplicate `_FALLBACK_CONTEXT_WINDOW` constant

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` (~line 511)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (~line 122)

**Changes:**
1. Move `_FALLBACK_CONTEXT_WINDOW = 128_000` to a shared location (e.g., `_lifecycle.py` as the canonical source)
2. Import it in `synthesis.py` instead of redefining

### 5D. Remove redundant `re` imports in supervision.py

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Remove inner `import re` statements in `_parse_combined_response` and `_extract_gap_analysis_section`
2. Module-level `import re` already exists at line 18

### 5E. Use `rfind` for supervisor brief splitting

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` (~lines 364-387)

**Changes:**
1. Replace `find("## SUPERVISOR BRIEF")` with `rfind("## SUPERVISOR BRIEF")`
2. Add check that `compressed` is non-empty after the split

---

## Dependency Graph

```
Phase 1 (Sanitization)  ← no dependencies, highest priority
Phase 2 (State Bugs)    ← no dependencies, high priority
Phase 3 (Validation)    ← no dependencies
Phase 4 (Tests)         ← after Phases 1-3 (tests verify the fixes)
Phase 5 (Performance)   ← no dependencies, can run in parallel with Phase 4
```

---

## Verification

After each phase, run:
```bash
pytest tests/core/research/ -x -q --tb=short
pytest tests/unit/ -x -q --tb=short
pytest tests/integration/test_deep_research_resilience.py -x -q --tb=short
```
