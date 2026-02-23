# PLAN-CHECKLIST: Align Compression with open_deep_research

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23

---

## Phase 1: Fix L2 Compression Input

- [x] **1.1** Rewrite compression to use full topic research context
  - [x] Include reflection notes, refined queries, completion rationale in prompt
  - [x] Include search iteration history (which query found which sources)
  - [x] Pass full source content instead of re-truncated snippets
- [x] **1.2** Raise source content limit to 50,000 chars (match open_deep_research)
  - [x] Add `deep_research_compression_max_content_length` to `ResearchConfig` (default 50000)
  - [x] Replace `_COMPRESSION_SOURCE_CHAR_LIMIT = 2000` class constant
- [x] **1.3** Align compression prompt with open_deep_research directives
  - [x] "DO NOT summarize — preserve verbatim, clean up format"
  - [x] "A later LLM will merge — don't lose sources"
  - [x] Output format: Queries Made → Comprehensive Findings → Source List
  - [x] Inline citations [1], [2] numbered sequentially
- [x] **1.4** Refactor to use `execute_llm_call` instead of duplicated retry logic
  - [x] Remove manual ContextWindowError retry loop from `compression.py`
  - [x] Use shared progressive token-limit recovery

---

## Phase 2: Fix L1 Summarization Input Limit

- [x] **2.1** Add `max_content_length` cap to `SourceSummarizer`
  - [x] Truncate content before building summarization prompt
  - [x] Default: 50,000 chars (matches open_deep_research)
- [x] **2.2** Add `deep_research_max_content_length` config field
  - [x] Wire through `_attach_source_summarizer` in `gathering.py`

---

## Phase 3: Update Tests

- [ ] **3.1** Update `test_topic_compression.py` for new prompt structure
  - [ ] Verify full ReAct context appears in compression prompt
  - [ ] Verify raised char limit is used
  - [ ] Verify prompt matches open_deep_research directives
- [ ] **3.2** Add input limit tests to `test_source_summarization.py`
  - [ ] Content exceeding max_content_length is truncated
  - [ ] Configurable limit is respected
- [ ] **3.3** Verify analysis consumes new compression format
  - [ ] Update `test_cross_phase_integration.py` if needed

---

## Phase 4: Cleanup from Code Review

- [ ] **4.1** Fix `_load_model_token_limits()` — use `foundry_mcp.config.__file__` not `parents[5]`
- [ ] **4.2** Extract `safe_resolve_model_for_role()` helper — replace 5+ try/except sites
- [ ] **4.3** Remove unused `provider_hint` from `_CONTEXT_WINDOW_ERROR_PATTERNS`
- [ ] **4.4** Add test: `_FALLBACK_MODEL_TOKEN_LIMITS` matches `model_token_limits.json`
- [ ] **4.5** Move per-call imports to module level in `_lifecycle.py`

---

## Sign-off

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Phase 1 | **Done** | 2026-02-23 | Core fix — compression input alignment |
| Phase 2 | **Done** | 2026-02-23 | L1 input cap |
| Phase 3 | Pending | — | Test updates |
| Phase 4 | Pending | — | Review cleanup |
