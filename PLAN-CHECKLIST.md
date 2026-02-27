# PLAN-CHECKLIST: Senior Review Findings Fix Plan

## Phase 1: Critical Fixes

- [x] **1a.** Fix token double-counting in parallel compression (CRITICAL)
  - [x] Add `skip_token_tracking` parameter to `execute_llm_call()` in `_lifecycle.py`
  - [x] Pass `skip_token_tracking=True` from `_compress_single_topic_async` in `compression.py`
  - [x] Verify `_apply_inline_compression_async` in `topic_research.py:1178` remains the sole token tracker
  - [x] Add test: parallel compression on 3 topics, assert `total_tokens_used` == sum (not 2x)

- [x] **1b.** Add logging on empty compression LLM response (CRITICAL)
  - [x] `compression.py:700-719` — add `logger.warning` + `_write_audit_event` before `return (0, 0, False)`
  - [x] Add test: mock LLM returning `success=True, content=""`, verify warning logged and audit event written

## Phase 2: Security Hardening

- [x] **2a.** Enable DNS resolution for extract URL validation (MAJOR)
  - [x] `topic_research.py:1647` — pass `resolve_dns=True` to `validate_extract_url`
  - [x] Add test: mock `socket.getaddrinfo` → `127.0.0.1`, verify URL rejected (covered by existing TestValidateExtractUrlDnsRebinding)

- [x] **2b.** Sanitize `supervisor_summary` and `per_topic_summary` at construction (MAJOR)
  - [x] `supervision.py:1353` — wrap `per_topic_summary` with `sanitize_external_content()`
  - [x] `supervision.py:1467` — wrap `supervisor_summary` with `sanitize_external_content()`
  - [x] Add test: injection tags in `per_topic_summary` stripped by `_build_directive_fallback_summary`

## Phase 3: Token Budget & Truncation Robustness

- [ ] **3a.** Guard against zero/negative token budgets (MAJOR)
  - [ ] `_token_budget.py:60-77` — add early return with warning when `max_tokens <= 0`
  - [ ] Add test: `truncate_to_token_estimate("text", 0)` returns full text + logs warning
  - [ ] Add test: `truncate_to_token_estimate("text", -5)` returns full text + logs warning

- [ ] **3b.** Account for `token_safety_margin` in supplementary notes injection (MAJOR)
  - [ ] `synthesis.py:1179-1190` — use `effective_context = context_window * (1 - safety_margin)`
  - [ ] Add test: headroom available with raw context but not with margin → injection skipped

- [ ] **3c.** Fix model token limit substring matching order (MAJOR)
  - [ ] `_model_resolution.py:21-41` — sort matches by descending key length (longest first)
  - [ ] Add test: dict with `"gpt-4": 8192` and `"gpt-4o": 128000`, model `"gpt-4o-2024"` → 128000

## Phase 4: Configuration Validation & Consistency

- [ ] **4a.** Validate phase-specific timeouts and thresholds (MAJOR)
  - [ ] `research.py` `_validate_deep_research_bounds()` — add lower-bound checks for all timeout fields
  - [ ] Validate `content_dedup_threshold` in [0, 1] range
  - [ ] Validate `compression_max_content_length` > 0
  - [ ] Add test: negative timeouts clamped to defaults with warnings

- [ ] **4b.** Fix `structured_truncate_blocks` single-pass insufficiency (MAJOR)
  - [ ] `_token_budget.py:177` — wrap in multi-pass loop (max 3 passes)
  - [ ] Add test: prompt 3x over budget with one large section → fits after truncation

## Phase 5: Test Quality & Dead Code Cleanup

- [ ] **5a.** Delete bogus wall-clock timeout tests
  - [ ] `test_sanitize_external_content.py:874-978` — delete `TestSupervisionWallClockTimeout` class (4 methods)

- [ ] **5b.** Remove dead `Contradiction` model class
  - [ ] `models/deep_research.py:609` — delete `Contradiction` class
  - [ ] Clean up any `contradictions` field references on `DeepResearchState`
  - [ ] Verify no production code imports `Contradiction`

- [ ] **5c.** Add test for compression hard-truncation mid-message
  - [ ] Test: message history with role markers, small `max_content_length`, verify truncation behavior documented

---

**Total items**: 25 checklist items across 5 phases
**Blocking items**: Phase 1 (2 items, CRITICAL) and Phase 2 (2 items, MAJOR/security) should be completed before merge
**Non-blocking**: Phases 3-5 can be batched or done in follow-up PRs
