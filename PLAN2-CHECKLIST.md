# PLAN2 — Post-Merge Improvements Checklist

Companion to [PLAN2.md](PLAN2.md). Each item is a discrete, actionable fix or follow-up.
Mark items `[x]` as completed.

---

## Phase 1 — Code Quality & Quick Fixes

- [x] **1.1** Standardize log levels for parsing/recovery events
  - Parsing fallbacks: DEBUG
  - Compression/recovery retries: WARNING
  - Hard failures: ERROR
  - Files: `gathering.py`, `_lifecycle.py`, `topic_research.py`, `clarification.py`
  - Only `clarification.py` needed changes: legacy parser fallbacks downgraded from WARNING/ERROR → DEBUG
- [x] **1.2** Document magic number rationale
  - `_MAX_TOKEN_LIMIT_RETRIES = 3`
  - `_TRUNCATION_FACTOR = 0.9`
  - `_COMPRESSION_SOURCE_CHAR_LIMIT = 2000`
  - `_FALLBACK_CONTEXT_WINDOW = 128_000`
- [x] **1.3** ~~Add context-window error patterns for Mistral and Cohere~~ **SKIPPED** — these providers are not used as LLM providers in deep research; patterns would be dead code
- [x] **1.4** Clean up compression token tracking in `gathering.py`
  - Replace `nonlocal` counter mutation with per-task result accumulation
  - `compress_one()` now returns `(input_tokens, output_tokens, success)` tuple
  - Aggregation happens after `asyncio.gather()` completes
- [x] **1.5** Replace O(n) citation scan with running counter in `DeepResearchState`
  - Added `next_citation_number: int` field with `model_validator` for backward compat
  - `add_source()` and `append_source()` now use O(1) counter increment

---

## Phase 2 — Test Coverage

- [x] **2.1** Add prompt content validation tests for compression and reflection
  - Added `test_compression_prompt_citation_numbering` — verifies [1], [2], [3] numbering
  - Added `test_compression_prompt_truncates_long_content` — verifies content truncation at char limit
  - Added `test_compression_prompt_source_count_header` — verifies source count in header
  - Added `test_prompt_includes_quality_distribution_counts` — verifies quality level counts
  - Added `test_prompt_includes_original_and_current_query` — verifies both queries in prompt
- [x] **2.2** Add concurrent state mutation stress tests
  - New file: `test_concurrent_state.py` (8 tests)
  - Token increment consistency (with and without lock)
  - Citation number uniqueness under concurrent append_source/add_source
  - URL deduplication under concurrent appends with lock
  - Multi-topic-agent simulation with shared state, overlapping URLs, high-contention tokens
- [x] **2.3** Add token recovery + downstream error combination tests
  - Added `test_truncation_succeeds_but_llm_returns_failure` — truncation fixes size but LLM returns success=False
  - Added `test_truncation_succeeds_but_llm_times_out` — truncation fixes size but LLM times out
  - Added `test_very_small_truncated_prompt_still_submitted` — tiny max_tokens budget, all retries exhausted
  - Added `test_non_context_error_after_successful_truncation` — RuntimeError after truncation propagates immediately
- [x] **2.4** Add `resolve_model_for_role()` edge case tests
  - Added `TestResolveModelForRoleEdgeCases` class (8 tests)
  - Empty config all defaults, empty string provider spec, malformed bracket specs
  - Role resolution failure gracefully handled in execute_llm_call
  - Model set without provider falls through to default
- [x] **2.5** Add cross-phase integration test
  - New file: `test_cross_phase_integration.py` (5 tests)
  - Full pipeline: clarification → planning → (simulated gathering) → synthesis
  - Constraint propagation from clarification to planning prompt
  - Findings and sources visible in synthesis prompt
  - Empty findings generates minimal report without LLM call
- [x] **2.6** Consolidate legacy `test_clarification.py`
  - Deleted `test_clarification.py`
  - Merged 13 unique edge-case tests into `test_clarification_structured.py` as `TestLegacyParseClarificationResponse`
  - Tests cover: question truncation, empty filtering, constraint value normalization, nested dict filtering, truthy/falsy coercion
  - Legacy tests clearly marked as safety net until Phase 3.3 removes legacy parser

---

## Phase 3 — Architecture Refactors

- [x] **3.1** Refactor `ResearchConfig` into nested sub-configs
  - Created `research_sub_configs.py` with `TavilyConfig`, `PerplexityConfig`, `SemanticScholarConfig`, `DeepResearchConfig`, `ModelRoleConfig` frozen dataclasses
  - Added `@property` accessors on `ResearchConfig`: `tavily_config`, `perplexity_config`, `semantic_scholar_config`, `deep_research_config`, `model_role_config`
  - Flat fields remain as source of truth — full backward compat, zero test changes
  - All 5898 tests pass
- [x] **3.2** Extract `_compress_topic_findings_async` from `GatheringPhaseMixin`
  - New file: `phases/compression.py` with `CompressionMixin` class
  - `GatheringPhaseMixin` now inherits from `CompressionMixin`
  - Exported from `phases/__init__.py`
  - All 49 compression tests + 432 gathering/integration tests pass
- [x] **3.3** Consolidate clarification parsing into single coherent path
  - Removed legacy `_parse_clarification_response()` method from `clarification.py`
  - Replaced with module-level `_extract_inferred_constraints()` pure function
  - Updated `TestLegacyParseClarificationResponse` → `TestExtractInferredConstraints` (10 tests)
  - Added 2 new edge-case tests (no constraints field, no JSON in content)
  - All 50 clarification tests pass
- [x] **3.4** Externalize `MODEL_TOKEN_LIMITS` to config
  - New file: `config/model_token_limits.json` with 19 model entries
  - `_lifecycle.py` loads from JSON at import time via `_load_model_token_limits()`
  - Hardcoded `_FALLBACK_MODEL_TOKEN_LIMITS` dict retained for resilience
  - Added 4 tests: loaded from JSON, fallback on missing file, fallback on malformed JSON, ordering preserved
  - All 22 lifecycle tests pass

---

## Phase 4 — Validation & Sign-off

- [ ] **4.1** Run full test suite (`pytest tests/`) — all tests pass
- [ ] **4.2** Run contract tests (`pytest tests/contract/`) — envelope schemas valid
- [ ] **4.3** Manual end-to-end test: deep research session with all features enabled
- [ ] **4.4** Compare token usage before/after on a reference query (document in PR)
- [ ] **4.5** Verify backward-compat: load pre-existing saved research session, confirm deserialization and resume
- [ ] **4.6** Review all new config fields have sensible defaults and documentation
