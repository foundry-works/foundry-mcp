# PLAN: Pre-Merge Review Findings — Round 2

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-26
**Status:** Draft

---

## Context

Second-round senior engineering review of the full branch (86 commits, +33.6k/-6.3k lines) after the Phase 1–6 hardening pass. The hardening fixed many issues, but this review identified remaining gaps — primarily prompt injection surface that was only partially addressed, config validation holes, state growth risks, and several robustness issues.

---

## Phase 1: Prompt Injection — Complete Sanitization Coverage

**Effort:** Medium | **Impact:** Critical
**Goal:** Apply `sanitize_external_content()` consistently across all phases where web-sourced data enters LLM prompts. Currently only `supervision.py` applies it.

### 1A. Sanitize topic researcher message history

**Problem:** `topic_research.py` — web-scraped content (search results, extracted pages) flows unsanitized into the ReAct conversation history via `_build_react_user_prompt`. The researcher LLM has tool-use capabilities, making this the highest-risk injection vector. An adversarial page could inject XML tags (`<system>`, `<assistant>`, `<invoke>`) to hijack the researcher.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

**Changes:**
1. Import `sanitize_external_content` from `_helpers`
2. In `_handle_web_search_tool` — sanitize each search result's title, snippet, and content before appending to `message_history`
3. In `_handle_extract_tool` — sanitize extracted page content before appending to `message_history`
4. In `_format_source_block` — sanitize `source.title`, `source.snippet`, `source.content` before formatting

### 1B. Sanitize synthesis prompt inputs

**Problem:** `synthesis.py` — `compressed_findings`, `raw_notes`, source titles/snippets are interpolated directly into LLM prompts without sanitization.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

**Changes:**
1. Import `sanitize_external_content` from `_helpers`
2. Sanitize `tr.compressed_findings` before interpolation into synthesis prompt
3. Sanitize `state.raw_notes` entries before joining into prompt
4. Sanitize source titles and snippets in the source-listing sections of the prompt

### 1C. Sanitize compression prompt inputs

**Problem:** `compression.py` — full `src.content`, `src.snippet`, `src.title` used unsanitized in compression prompts.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Changes:**
1. Import `sanitize_external_content` from `_helpers`
2. Sanitize all source-derived content (`content`, `snippet`, `title`) before interpolation into compression prompts
3. Sanitize raw notes content before prompt inclusion

### 1D. Sanitize brief and evaluation prompt inputs

**Problem:** `brief.py` and `evaluation/evaluator.py` also interpolate source-derived content into prompts without sanitization.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`
- `src/foundry_mcp/core/research/evaluation/evaluator.py`

**Changes:**
1. Import `sanitize_external_content` in both files
2. In `brief.py` — sanitize any source content that feeds into the brief generation prompt
3. In `evaluator.py` — sanitize source titles, raw notes, and any web-derived content in the evaluation prompt

### 1E. Fix Python `.format()` injection in summarization prompt

**Problem:** `providers/shared.py:789` uses `_SOURCE_SUMMARIZATION_PROMPT.format(content=content)`. Raw web content containing Python format string patterns (`{system}`, `{__class__}`) could raise `KeyError` or expose internals.

**Files:**
- `src/foundry_mcp/core/research/providers/shared.py`

**Changes:**
1. Replace `.format(content=content)` with a safer mechanism — either `string.Template` with `safe_substitute`, or manual string concatenation
2. Also apply `sanitize_external_content()` to the content before interpolation

### 1F. Add cross-phase sanitization tests

**Files:**
- `tests/core/research/workflows/deep_research/test_sanitize_external_content.py` (extend existing)

**Changes:**
1. Add tests verifying that injection payloads in search results are sanitized before reaching the topic researcher LLM
2. Add tests verifying synthesis prompt sanitization
3. Add tests verifying compression prompt sanitization
4. Test that `.format()` injection in summarization content is handled safely

---

## Phase 2: Config Validation & Backward Compatibility

**Effort:** Low | **Impact:** High
**Goal:** Add validation for new supervision config fields and deprecation warnings for removed fields.

### 2A. Add supervision config validation

**Problem:** `deep_research_max_supervision_rounds` has no upper bound. A user setting `= 1000` would execute hundreds of delegation rounds with massive LLM API cost.

**Files:**
- `src/foundry_mcp/config/research.py`

**Changes:**
1. Add `_validate_supervision_config()` method called from `__post_init__`
2. Cap `deep_research_max_supervision_rounds` at 20 (warn + clamp)
3. Validate `deep_research_coverage_confidence_threshold` is in [0.0, 1.0]
4. Validate `deep_research_max_concurrent_research_units` has a sane upper bound (e.g., 20)

### 2B. Add deprecation warnings for removed config fields

**Problem:** Several fields were removed (`deep_research_enable_reflection`, `deep_research_enable_contradiction_detection`, `deep_research_digest_policy`, `tavily_extract_in_deep_research`, `tavily_extract_max_urls`, analysis/refinement provider/timeout fields). Users with existing TOML configs won't know their settings no longer apply.

**Files:**
- `src/foundry_mcp/config/research.py`

**Changes:**
1. Define `_DEPRECATED_FIELDS` set with all removed field names
2. In `from_dict()` / `from_toml_dict()`, check input data keys against `_DEPRECATED_FIELDS`
3. Log `warnings.warn(..., DeprecationWarning)` for each deprecated field found
4. Include migration hint (e.g., "deep_research_enable_reflection has been removed; reflection is now always-on in the supervision loop")

### 2C. Fix ambiguous model name in cost tier defaults

**Problem:** `_COST_TIER_MODEL_DEFAULTS` uses `"2.0-flash"` which doesn't match entries in `model_token_limits.json` (which uses `"gemini-2.5-flash"` style). Token limit estimation falls back to default.

**Files:**
- `src/foundry_mcp/config/research.py`

**Changes:**
1. Update `_COST_TIER_MODEL_DEFAULTS` values to use full model names matching `model_token_limits.json`
2. Or add `"2.0-flash"` entries to `model_token_limits.json` for match consistency

---

## Phase 3: State Growth & Resource Guards

**Effort:** Medium | **Impact:** High
**Goal:** Prevent unbounded state growth and add wall-clock guards to the supervision loop.

### 3A. Cap `raw_notes` in `DeepResearchState`

**Problem:** Each topic researcher appends full raw notes to `state.raw_notes`. With 6 rounds × 5 researchers, this can grow to MBs. `supervision_messages` has truncation via `truncate_supervision_messages()`; `raw_notes` has none.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Add `_MAX_RAW_NOTES = 50` constant (or character budget, e.g., 500K chars)
2. After appending raw notes, trim oldest entries if count/size exceeds cap
3. Add audit log when trimming occurs

### 3B. Add wall-clock timeout guard to supervision phase

**Problem:** The supervision loop runs up to `max_supervision_rounds` iterations with no overall wall-clock guard. Each round can spawn many LLM calls + parallel researchers. A slow provider could make a single round take minutes.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Record `phase_start = time.monotonic()` at the start of `execute_supervision_phase`
2. Add `max_phase_wall_clock` config field (default 1800s = 30 minutes)
3. Check elapsed time at the top of each round iteration; break with warning if exceeded
4. Log the early exit reason in state metadata

### 3C. Remove double `asyncio.wait_for` in compression

**Problem:** `_compress_directive_results_inline` wraps `_compress_single_topic_async` (which already takes `timeout`) in `asyncio.wait_for(timeout=compression_timeout)`. Double timeout wrapping is redundant and the outer cancellation could leave resources inconsistent.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Remove the outer `asyncio.wait_for` wrapper
2. Rely on the inner timeout mechanism within `_compress_single_topic_async`
3. Or vice versa — remove the inner timeout and rely on the outer `wait_for`

---

## Phase 4: Robustness & Correctness Fixes

**Effort:** Low | **Impact:** Medium
**Goal:** Fix edge cases in phase advancement, authorization, dedup, and parsing.

### 4A. Make `advance_phase()` skip logic handle consecutive deprecated phases

**Problem:** Current logic hardcodes a `+2` skip. If two consecutive phases are ever deprecated, only the first is skipped. The bounds check also has a subtle off-by-one.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`

**Changes:**
1. Replace the single-skip logic with a while loop:
   ```python
   while self.phase in self._SKIP_PHASES and current_index < len(phase_order) - 1:
       current_index += 1
       self.phase = phase_order[current_index]
   ```
2. Add test for consecutive-skip scenario

### 4B. Add `deep-research-evaluate` to authorization allowlists

**Problem:** The new `deep-research-evaluate` action is a read-only analysis but is missing from `OBSERVER_ALLOWLIST` and `AUTONOMY_RUNNER_ALLOWLIST`. Only maintainers can use it.

**Files:**
- `src/foundry_mcp/core/authorization.py`

**Changes:**
1. Add `"deep-research-evaluate"` to `OBSERVER_ALLOWLIST` (it's a read-only analysis action)
2. Consider adding to `AUTONOMY_RUNNER_ALLOWLIST` if runners need quality-gate evaluation

### 4C. Harden `_critique_has_issues` parsing

**Problem:** Checks for exact string `"VERDICT: NO_ISSUES"`. Minor LLM formatting variation (extra spaces, different casing of underscore) breaks parsing.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Replace exact string match with regex: `re.search(r"VERDICT\s*:\s*NO[_\s]?ISSUES", text_upper)`
2. Tighten the `"ISSUE:"` fallback check to avoid false positives on conversational uses of "issue"

### 4D. Verify dedup lock coverage for `seen_urls`/`seen_titles`

**Problem:** In `_execute_directives_async`, `seen_urls` and `seen_titles` are shared across parallel researchers. The check-then-add pattern for dedup may not be atomic under the `state_lock`.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Audit `_topic_search` to confirm `state_lock` is held across the check-and-add for `seen_urls`
2. If not, move the dedup check inside the lock-holding section
3. Add a test exercising concurrent researcher dedup

### 4E. Add researcher conversation history truncation

**Problem:** `_build_react_user_prompt` encodes full conversation history into the user prompt. With 30 turns of tool results containing web content, the prompt can exceed context windows on smaller models.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

**Changes:**
1. Add a `_truncate_researcher_history` function (similar to `truncate_supervision_messages`)
2. Estimate token budget based on model context window
3. Preserve recent turns, summarize or drop older ones when over budget
4. Call before `_build_react_user_prompt` in the ReAct loop

### 4F. Sort model token limits by longest match

**Problem:** `model_token_limits.json` relies on JSON key ordering for first-match substring matching. New entries could silently change which limits are resolved.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`

**Changes:**
1. After loading the JSON, sort entries by key length descending (longest match first)
2. This makes the match order deterministic regardless of JSON key ordering

---

## Phase 5: Test Coverage for New Findings

**Effort:** Low-Medium | **Impact:** Medium
**Goal:** Add regression tests for all fixes in this plan.

### 5A. Sanitization coverage tests (extends Phase 1F)

Add tests confirming injection payloads are stripped across all phases, not just supervision.

### 5B. Config validation tests

Test that `max_supervision_rounds > 20` is clamped, `coverage_confidence_threshold` outside [0,1] raises, and deprecated fields produce warnings.

### 5C. State growth tests

Test that `raw_notes` is capped after many appends, and that supervision phase respects wall-clock timeout.

### 5D. Phase advancement tests

Test consecutive deprecated phases are all skipped correctly.

### 5E. Authorization tests

Test that `deep-research-evaluate` is allowed for observers.
