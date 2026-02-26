# PLAN-CHECKLIST: Pre-Merge Review Findings — Round 2

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-26

---

## Phase 1: Prompt Injection — Complete Sanitization Coverage

- [x] **1A.1** Import `sanitize_external_content` in `topic_research.py`
- [x] **1A.2** Sanitize search result titles/snippets/content in `_handle_web_search_tool` before appending to `message_history`
- [x] **1A.3** Sanitize extracted page content in `_handle_extract_tool` before appending to `message_history`
- [x] **1A.4** Sanitize source fields in `_format_source_block`
- [x] **1B.1** Import `sanitize_external_content` in `synthesis.py`
- [x] **1B.2** Sanitize `tr.compressed_findings` before interpolation into synthesis prompt
- [x] **1B.3** Sanitize `state.raw_notes` entries before joining into prompt
- [x] **1B.4** Sanitize source titles/snippets in synthesis source-listing sections
- [x] **1C.1** Import `sanitize_external_content` in `compression.py`
- [x] **1C.2** Sanitize all source-derived content (`content`, `snippet`, `title`) in compression prompts
- [x] **1C.3** Sanitize raw notes content in compression prompts
- [N/A] **1D.1** Import `sanitize_external_content` in `brief.py` — brief phase uses only user query, no web-derived content
- [N/A] **1D.2** Sanitize source content feeding into brief generation prompt — not applicable (no sources in brief)
- [x] **1D.3** Import `sanitize_external_content` in `evaluation/evaluator.py`
- [x] **1D.4** Sanitize source titles, raw notes, and web-derived content in evaluation prompt
- [x] **1E.1** Replace `.format(content=content)` in `providers/shared.py` with safe interpolation (e.g., `string.Template.safe_substitute` or concatenation)
- [x] **1E.2** Apply `sanitize_external_content()` to content before prompt interpolation in `shared.py`
- [x] **1F.1** Add test: injection payload in search results is stripped before reaching topic researcher LLM
- [x] **1F.2** Add test: injection payload in source content is stripped in synthesis prompt
- [x] **1F.3** Add test: injection payload in source content is stripped in compression prompt
- [x] **1F.4** Add test: `.format()` injection in summarization content is handled safely

---

## Phase 2: Config Validation & Backward Compatibility

- [x] **2A.1** Add `_validate_supervision_config()` method to `ResearchConfig.__post_init__`
- [x] **2A.2** Cap `deep_research_max_supervision_rounds` at 20 (warn + clamp)
- [x] **2A.3** Validate `deep_research_coverage_confidence_threshold` in [0.0, 1.0]
- [x] **2A.4** Validate `deep_research_max_concurrent_research_units` upper bound (e.g., 20)
- [x] **2B.1** Define `_DEPRECATED_FIELDS` set with all removed field names in `research.py`
- [x] **2B.2** Add deprecation check in `from_dict()` / `from_toml_dict()` — log `DeprecationWarning` for each match
- [x] **2B.3** Include migration hints in deprecation messages
- [x] **2C.1** Update `_COST_TIER_MODEL_DEFAULTS` to use full model names matching `model_token_limits.json`

---

## Phase 3: State Growth & Resource Guards

- [x] **3A.1** Add `_MAX_RAW_NOTES` constant to `supervision.py` (count or char budget)
- [x] **3A.2** Trim oldest `raw_notes` entries when cap exceeded after appending
- [x] **3A.3** Add audit log entry when raw notes are trimmed
- [x] **3B.1** Record `phase_start = time.monotonic()` at start of `execute_supervision_phase`
- [x] **3B.2** Add `max_phase_wall_clock` config (default 1800s)
- [x] **3B.3** Check elapsed time at top of each supervision round; break with warning if exceeded
- [x] **3B.4** Log early exit reason in state metadata
- [x] **3C.1** Remove outer `asyncio.wait_for` in `_compress_directive_results_inline` (or inner timeout) — use one mechanism, not both

---

## Phase 4: Robustness & Correctness Fixes

- [ ] **4A.1** Replace `advance_phase()` single-skip with while loop over `_SKIP_PHASES`
- [ ] **4A.2** Add test for consecutive deprecated phases being skipped
- [ ] **4B.1** Add `"deep-research-evaluate"` to `OBSERVER_ALLOWLIST`
- [ ] **4B.2** Consider adding to `AUTONOMY_RUNNER_ALLOWLIST` if runners need quality gates
- [ ] **4C.1** Replace exact `"VERDICT: NO_ISSUES"` match with regex `r"VERDICT\s*:\s*NO[_\s]?ISSUES"` in `_critique_has_issues`
- [ ] **4C.2** Tighten `"ISSUE:"` fallback check to reduce false positives
- [ ] **4D.1** Audit `_topic_search` — confirm `state_lock` held across check-and-add for `seen_urls`
- [ ] **4D.2** If not atomic, move dedup check inside lock-holding section
- [ ] **4D.3** Add test exercising concurrent researcher dedup
- [ ] **4E.1** Add `_truncate_researcher_history` function to `topic_research.py`
- [ ] **4E.2** Estimate token budget based on model context window
- [ ] **4E.3** Call truncation before `_build_react_user_prompt` in the ReAct loop
- [ ] **4F.1** Sort model token limits by key length descending after loading JSON

---

## Phase 5: Test Coverage for New Findings

- [ ] **5A.1** Extend `test_sanitize_external_content.py` with cross-phase injection scenarios
- [x] **5B.1** Test: `max_supervision_rounds = 100` is clamped to 20
- [x] **5B.2** Test: `coverage_confidence_threshold = 1.5` raises or is clamped
- [x] **5B.3** Test: deprecated field in TOML input produces `DeprecationWarning`
- [ ] **5C.1** Test: `raw_notes` list is capped after many appends
- [ ] **5C.2** Test: supervision phase respects wall-clock timeout and exits early
- [ ] **5D.1** Test: consecutive deprecated phases are all skipped by `advance_phase()`
- [ ] **5E.1** Test: `deep-research-evaluate` action is allowed for observer role

---

## Final Validation

- [ ] Full deep research test suite passes: `python -m pytest tests/core/research/ -x -q`
- [ ] Evaluation tests pass: `python -m pytest tests/core/research/evaluation/ -x -q`
- [ ] Integration tests pass: `python -m pytest tests/integration/ -x -q`
- [ ] No regressions in existing workflow tests
- [ ] Grep confirms no remaining unsanitized `source.content` or `source.title` in prompt-building code paths
