# PLAN1 — Review Findings Checklist

Companion to [PLAN1.md](PLAN1.md). Each item is a discrete, actionable fix or follow-up.
Mark items `[x]` as completed.

---

## Pre-Merge: Must Fix

- [x] **PM.1** Replace `assert result is not None` with explicit error check in `_lifecycle.py:379`
  - Return `WorkflowResult(success=False, ...)` instead of asserting
  - Assertions are disabled by `python -O` and should not guard user-facing code paths
- [x] **PM.2** Fix deprecated `datetime.utcnow()` in `models/fidelity.py`
  - Replace with `datetime.now(timezone.utc)` to match the rest of the codebase
  - `utcnow()` is deprecated since Python 3.12
- [x] **PM.3** Add DEBUG logging to silent role-resolution `except` blocks (4 locations)
  - `gathering.py` lines 122-125 (`_compress_topic_findings_async` role resolution)
  - `gathering.py` `_attach_source_summarizer` role resolution
  - `_lifecycle.py` `execute_llm_call` role resolution
  - `topic_research.py` reflection provider resolution
  - Pattern: `except (AttributeError, TypeError, ValueError): logger.debug("Role resolution unavailable for %s, using defaults", role)`
- [x] **PM.4** Run full test suite (`pytest tests/`) — 5885 passed, 48 skipped, 0 failures
- [x] **PM.5** Run contract tests (`pytest tests/contract/`) — 42 passed, 0 failures

---

## Pre-Merge: Should Fix

- [x] **PS.1** Overhaul `MODEL_TOKEN_LIMITS` in `_lifecycle.py` with current models
  - Removed outdated entries (claude-3/3.5, gpt-3.5-turbo, gpt-4, o3/o4, gemini-1.5/2.x)
  - Added: claude-opus-4-6/sonnet-4-6/haiku-4-5, gpt-5.3-codex/spark, gpt-5.1-codex-mini, gpt-5-mini, gpt-4.1/mini, gemini-3.x pro/flash
  - Verified ordering: more-specific substrings precede less-specific throughout
- [x] **PS.2** Removed unused `allocated_map` variable in `_analysis_prompts.py`
  - Variable was built but never referenced; dead code removed
- [x] **PS.3** Renamed `_provider_hint` to `_` in `_is_context_window_error` loop
  - Variable is unpacked but intentionally unused; uses `_` for clarity
- [x] **PS.4** Validated `MODEL_TOKEN_LIMITS` ordering for substring matching
  - Fixed: `"claude-3.5"` before `"claude-3"` (was reversed) — moot now, both removed
  - Verified all specific IDs precede generic family substrings throughout

---

## Post-Merge: Architecture Follow-Up

- [ ] **PA.1** Refactor `ResearchConfig` into nested sub-configs
  - Extract `TavilyConfig`, `PerplexityConfig`, `DeepResearchConfig`, `ModelRoleConfig`
  - Reduce field count from ~240 to manageable groups
  - Maintain backward compatibility via property accessors
- [ ] **PA.2** Extract `_compress_topic_findings_async` from `GatheringPhaseMixin`
  - Move to dedicated `CompressionPhaseMixin` or utility module
  - Current 270-line method is too large for a mixin method
- [ ] **PA.3** Consolidate clarification parsing into single coherent path
  - Current: structured → lenient → legacy `_parse_clarification_response()`
  - Target: structured → lenient fallback (remove legacy path)
  - Verify `inferred_constraints` extraction works with new schema before removing legacy
- [ ] **PA.4** Externalize `MODEL_TOKEN_LIMITS` to config
  - Either load from a config file or query provider capabilities at startup
  - Current hardcoded dict requires code changes for new models
- [ ] **PA.5** Replace O(n) citation scan with running counter in `DeepResearchState`
  - `add_source()` currently scans all sources for max citation number
  - Maintain `_next_citation_number: int` on the state object

---

## Post-Merge: Test Improvements

- [ ] **PT.1** Add prompt content validation tests for compression and reflection
  - Verify compression prompt includes correct source content, URLs, citations
  - Verify reflection prompt includes source count and quality distribution
  - Currently mocked but actual prompt structure unchecked
- [ ] **PT.2** Add concurrent state mutation stress tests
  - Test `state_lock` correctness with multiple topic agents
  - Test `total_tokens_used` consistency under concurrent updates
- [ ] **PT.3** Add token recovery + downstream error combination test
  - What happens if truncation succeeds but the LLM still fails?
  - What if truncated prompt is too small to produce useful output?
- [ ] **PT.4** Add `resolve_model_for_role()` edge case tests
  - Empty config (no fields set at all)
  - Role set but provider resolution returns None
  - Malformed provider specs (`[]model`, `[provider]`, etc.)
- [ ] **PT.5** Add cross-phase integration test
  - End-to-end: clarification → planning → gathering → compression → analysis → synthesis
  - Verify state propagation and data consistency across phases
- [ ] **PT.6** Mark `test_clarification.py` as legacy or consolidate
  - Tests old `needs_clarification` schema (plural)
  - New code uses `need_clarification` (singular)
  - Either rename to `test_clarification_legacy.py` or merge relevant tests into `test_clarification_structured.py`

---

## Post-Merge: Code Quality

- [ ] **PQ.1** Standardize log levels for parsing/recovery events
  - Parsing fallbacks: DEBUG
  - Compression/recovery retries: WARNING
  - Hard failures: ERROR
  - Currently inconsistent (some parsing failures at INFO, others at WARNING)
- [ ] **PQ.2** Document magic number rationale
  - `_MAX_TOKEN_LIMIT_RETRIES = 3` — why 3?
  - `_TRUNCATION_FACTOR = 0.9` — why 10% reduction?
  - `_COMPRESSION_SOURCE_CHAR_LIMIT = 2000` — why 2000 chars?
  - `_FALLBACK_CONTEXT_WINDOW = 128_000` — why 128K?
  - Add brief comments explaining the reasoning or link to design docs
- [ ] **PQ.3** Add context-window error patterns for additional providers
  - Current coverage: OpenAI, Anthropic, Google
  - Missing: Mistral, Cohere, other providers used via provider framework
- [ ] **PQ.4** Clean up compression token tracking pattern
  - Replace `nonlocal` counter mutation with per-task result accumulation
  - Sum results after `asyncio.gather()` completes for cleaner concurrency

---

## Cross-Cutting Validation (from PLAN-CHECKLIST.md)

- [ ] **V.1** Run full test suite (`pytest tests/`) — all existing tests pass
- [ ] **V.2** Run contract tests (`pytest tests/contract/`) — envelope schemas valid
- [ ] **V.3** Manual end-to-end test: run a deep research session with all features enabled
- [ ] **V.4** Compare token usage before/after on a reference query (document in PR)
- [ ] **V.5** Verify backward-compat: load pre-existing saved research session, confirm deserialization and resume
- [ ] **V.6** Review all new config fields have sensible defaults and documentation
