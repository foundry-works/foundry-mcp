# PLAN2 — Post-Merge Improvements Checklist

Companion to [PLAN2.md](PLAN2.md). Each item is a discrete, actionable fix or follow-up.
Mark items `[x]` as completed.

---

## Phase 1 — Code Quality & Quick Fixes

- [ ] **1.1** Standardize log levels for parsing/recovery events
  - Parsing fallbacks: DEBUG
  - Compression/recovery retries: WARNING
  - Hard failures: ERROR
  - Files: `gathering.py`, `_lifecycle.py`, `topic_research.py`, `clarification.py`
- [ ] **1.2** Document magic number rationale
  - `_MAX_TOKEN_LIMIT_RETRIES = 3`
  - `_TRUNCATION_FACTOR = 0.9`
  - `_COMPRESSION_SOURCE_CHAR_LIMIT = 2000`
  - `_FALLBACK_CONTEXT_WINDOW = 128_000`
- [ ] **1.3** Add context-window error patterns for Mistral and Cohere
  - Research their error message formats
  - Add regex entries to `_CONTEXT_WINDOW_ERROR_PATTERNS` in `_lifecycle.py`
- [ ] **1.4** Clean up compression token tracking in `gathering.py`
  - Replace `nonlocal` counter mutation with per-task result accumulation
  - Sum results after `asyncio.gather()` completes
- [ ] **1.5** Replace O(n) citation scan with running counter in `DeepResearchState`
  - Add `_next_citation_number: int` field
  - Increment on each `add_source()` call

---

## Phase 2 — Test Coverage

- [ ] **2.1** Add prompt content validation tests for compression and reflection
  - Verify compression prompt includes source content, URLs, citations
  - Verify reflection prompt includes source count and quality distribution
- [ ] **2.2** Add concurrent state mutation stress tests
  - Test `state_lock` correctness with multiple topic agents
  - Test `total_tokens_used` consistency under concurrent updates
- [ ] **2.3** Add token recovery + downstream error combination tests
  - Truncation succeeds but LLM still fails
  - Truncated prompt too small to produce useful output
- [ ] **2.4** Add `resolve_model_for_role()` edge case tests
  - Empty config (no fields set)
  - Role set but provider resolution returns None
  - Malformed provider specs (`[]model`, `[provider]`, empty string)
- [ ] **2.5** Add cross-phase integration test
  - End-to-end: clarification → planning → gathering → compression → analysis → synthesis
  - Verify state propagation and data consistency across phases
- [ ] **2.6** Consolidate legacy `test_clarification.py`
  - Merge relevant tests into `test_clarification_structured.py`
  - Or rename to `test_clarification_legacy.py` with explanatory comment

---

## Phase 3 — Architecture Refactors

- [ ] **3.1** Refactor `ResearchConfig` into nested sub-configs
  - Extract `TavilyConfig`, `PerplexityConfig`, `DeepResearchConfig`, `ModelRoleConfig`
  - Maintain backward compat via property accessors
  - Existing tests must pass without modification
- [ ] **3.2** Extract `_compress_topic_findings_async` from `GatheringPhaseMixin`
  - Move to dedicated `CompressionMixin` or utility module
  - 270-line method → standalone, independently testable unit
- [ ] **3.3** Consolidate clarification parsing into single coherent path
  - Remove legacy `_parse_clarification_response()` after verifying `inferred_constraints` works with new schema
  - Depends on: 2.6 (legacy test consolidation)
- [ ] **3.4** Externalize `MODEL_TOKEN_LIMITS` to config
  - Load from TOML/JSON config file or query provider capabilities
  - Remove hardcoded dict from `_lifecycle.py`

---

## Phase 4 — Validation & Sign-off

- [ ] **4.1** Run full test suite (`pytest tests/`) — all tests pass
- [ ] **4.2** Run contract tests (`pytest tests/contract/`) — envelope schemas valid
- [ ] **4.3** Manual end-to-end test: deep research session with all features enabled
- [ ] **4.4** Compare token usage before/after on a reference query (document in PR)
- [ ] **4.5** Verify backward-compat: load pre-existing saved research session, confirm deserialization and resume
- [ ] **4.6** Review all new config fields have sensible defaults and documentation
