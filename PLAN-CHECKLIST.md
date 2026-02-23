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

- [x] **3.1** Update `test_topic_compression.py` for new prompt structure
  - [x] Verify full ReAct context appears in compression prompt (TestFullReActContext: 4 tests)
  - [x] Verify raised char limit is used (configurable_content_limit + default_allows_long_content)
  - [x] Verify prompt matches open_deep_research directives (system_prompt_aligned test)
- [x] **3.2** Add input limit tests to `test_source_summarization.py`
  - [x] Content exceeding max_content_length is truncated (3 async tests)
  - [x] Configurable limit is respected (custom_limit_is_respected)
  - [x] Config field tests: default, explicit, TOML parsing, TOML default (4 tests)
- [x] **3.3** Verify analysis consumes new compression format
  - [x] Compressed findings format (Queries Made / Findings / Source List) in analysis prompt
  - [x] Inline citations preserved through analysis
  - [x] Source ID mapping included per topic
  - [x] Compressed findings flow through to synthesis (5 tests in TestCompressedFindingsCrossPhase)

---

## Phase 4: Cleanup from Code Review

- [x] **4.1** Fix `_load_model_token_limits()` — use `foundry_mcp.config.__file__` not `parents[5]`
  - [x] Import `foundry_mcp.config` as `_config_pkg`, resolve path via `Path(_config_pkg.__file__).resolve().parent`
  - [x] Updated fallback tests to mock `_config_pkg` instead of fragile `parents[5]` chain
- [x] **4.2** Extract `safe_resolve_model_for_role()` helper — replace 4 try/except sites
  - [x] Added `safe_resolve_model_for_role()` to `_helpers.py` — returns `(None, None)` on failure
  - [x] Replaced try/except in `_lifecycle.py`, `gathering.py`, `compression.py`, `topic_research.py`
- [x] **4.3** Remove unused `provider_hint` from `_CONTEXT_WINDOW_ERROR_PATTERNS`
  - [x] Simplified from `list[tuple[str, str]]` to `list[str]` (comments document provider coverage)
  - [x] Updated `_is_context_window_error()` loop and test assertions
- [x] **4.4** Add test: `_FALLBACK_MODEL_TOKEN_LIMITS` matches `model_token_limits.json`
  - [x] `test_fallback_matches_json` loads both sources and asserts equality with clear error message
- [x] **4.5** Move per-call imports to module level in `_lifecycle.py`
  - [x] Moved `estimate_token_limit_for_model`, `truncate_to_token_estimate`, `safe_resolve_model_for_role` to top-level imports
  - [x] Replaced `import json as _json` with module-level `json` (already imported)

---

## Sign-off

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Phase 1 | **Done** | 2026-02-23 | Core fix — compression input alignment |
| Phase 2 | **Done** | 2026-02-23 | L1 input cap |
| Phase 3 | **Done** | 2026-02-23 | 12 new tests (7 summarization + 5 cross-phase) |
| Phase 4 | **Done** | 2026-02-23 | Review cleanup — 5 items |
