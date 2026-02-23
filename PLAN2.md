# PLAN2 — Post-Merge Improvements: Deep Research Enhancements

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Predecessor:** [PLAN1.md](PLAN1.md) (pre-merge fixes — all complete)
**Scope:** 21 items across code quality, testing, architecture, and validation
**Date:** 2026-02-23

---

## Overview

PLAN1 addressed all pre-merge blockers and should-fix items. This plan organizes the remaining post-merge improvements into four phases, ordered by risk and dependency:

| Phase | Focus | Items | Risk | Dependency |
|-------|-------|-------|------|------------|
| 1 | Code quality & quick fixes | 5 | Low | None |
| 2 | Test coverage gaps | 6 | Low | None (but informing Phase 3) |
| 3 | Architecture refactors | 4 | Medium | Phase 2 tests provide safety net |
| 4 | Validation & sign-off | 6 | Low | Phases 1–3 complete |

---

## Phase 1 — Code Quality & Quick Fixes

Small, isolated, low-risk changes that improve readability and maintainability. Each can be done independently.

### 1.1 Standardize log levels (PQ.1)

**Problem:** Inconsistent log levels across parsing/recovery code paths — some parsing failures log at INFO, others at WARNING.

**Target convention:**
- Parsing fallbacks → DEBUG
- Compression/recovery retries → WARNING
- Hard failures → ERROR

**Files:** `gathering.py`, `_lifecycle.py`, `topic_research.py`, `clarification.py`

### 1.2 Document magic number rationale (PQ.2)

**Problem:** Several magic numbers lack explanation of why those specific values were chosen.

**Constants to document:**
- `_MAX_TOKEN_LIMIT_RETRIES = 3`
- `_TRUNCATION_FACTOR = 0.9`
- `_COMPRESSION_SOURCE_CHAR_LIMIT = 2000`
- `_FALLBACK_CONTEXT_WINDOW = 128_000`

**Action:** Add inline comments explaining the reasoning or link to design docs.

### 1.3 Add context-window error patterns for additional providers (PQ.3)

**Problem:** `_CONTEXT_WINDOW_ERROR_PATTERNS` only covers OpenAI, Anthropic, and Google. Other providers used via the provider framework (Mistral, Cohere, etc.) will not trigger progressive token recovery.

**Action:** Research error message patterns for Mistral and Cohere, add regex entries to `_CONTEXT_WINDOW_ERROR_PATTERNS` in `_lifecycle.py`.

### 1.4 Clean up compression token tracking (PQ.4)

**Problem:** `_compress_topic_findings_async` uses `nonlocal` counters mutated from concurrent coroutines within `asyncio.gather`. While safe under CPython's cooperative scheduling, the pattern is fragile and hard to reason about.

**Target:** Accumulate results per-task, sum after `gather()` completes.

**File:** `gathering.py`

### 1.5 Replace O(n) citation scan with running counter (PA.5)

**Problem:** `add_source()` in `DeepResearchState` scans all sources for `max(citation_number)` on every call — O(n) per add, O(n²) for n sources.

**Action:** Add `_next_citation_number: int` field to `DeepResearchState`, increment on each `add_source()` call.

**File:** `models/deep_research.py`

---

## Phase 2 — Test Coverage

Fill identified test gaps. Having comprehensive tests before Phase 3 refactors provides a safety net for architectural changes.

### 2.1 Prompt content validation tests (PT.1)

**Gap:** Compression and reflection prompts are mocked in tests but the actual prompt structure is never validated.

**Tests to add:**
- Verify compression prompt includes correct source content, URLs, and citations
- Verify reflection prompt includes source count and quality distribution

### 2.2 Concurrent state mutation stress tests (PT.2)

**Gap:** `state_lock` correctness and `total_tokens_used` consistency under concurrent updates are under-tested.

**Tests to add:**
- Multiple topic agents mutating state concurrently
- Token counter consistency after parallel updates

### 2.3 Token recovery + downstream error combination (PT.3)

**Gap:** No test for what happens when truncation succeeds but the LLM still fails, or when the truncated prompt is too small.

**Tests to add:**
- Truncation succeeds → LLM returns error anyway
- Truncated prompt below useful threshold

### 2.4 `resolve_model_for_role()` edge cases (PT.4)

**Gap:** Missing tests for empty/null config, provider resolution returning None, and malformed provider specs.

**Tests to add:**
- Empty config (no fields set)
- Role set but provider resolution returns None
- Malformed specs: `[]model`, `[provider]`, empty string

### 2.5 Cross-phase integration test (PT.5)

**Gap:** No end-to-end test covering the full pipeline: clarification → planning → gathering → compression → analysis → synthesis.

**Action:** Create a single integration test that verifies state propagation and data consistency across all phases. Use mocked providers but real phase logic.

### 2.6 Consolidate legacy clarification tests (PT.6)

**Problem:** `test_clarification.py` tests the old `needs_clarification` (plural) schema while new code uses `need_clarification` (singular).

**Options:**
- Rename to `test_clarification_legacy.py` with a comment explaining it tests backward compat
- Merge relevant tests into `test_clarification_structured.py` and delete the old file

---

## Phase 3 — Architecture Refactors

Larger structural changes. Phase 2 tests should be in place first to catch regressions.

### 3.1 Refactor `ResearchConfig` into nested sub-configs (PA.1)

**Problem:** `ResearchConfig` dataclass has ~240+ fields and growing. Phase 6 added 12 more role-based config fields.

**Target structure:**
```
ResearchConfig
├── TavilyConfig (search API settings)
├── PerplexityConfig (search API settings)
├── DeepResearchConfig (phase settings, timeouts, concurrency)
└── ModelRoleConfig (role-based model routing)
```

**Constraints:**
- Maintain backward compatibility via property accessors on `ResearchConfig`
- Existing tests must pass without modification (or with minimal fixture updates)

### 3.2 Extract compression from GatheringPhaseMixin (PA.2)

**Problem:** `_compress_topic_findings_async` is a 270-line method on `GatheringPhaseMixin`. Compression is a distinct operation that deserves its own module.

**Target:** Dedicated `CompressionMixin` or standalone utility module.

### 3.3 Consolidate clarification parsing (PA.3)

**Problem:** Three-layer parsing chain: structured → lenient → legacy `_parse_clarification_response()`. The legacy path extracts `inferred_constraints` from a schema that doesn't formally include them.

**Target:** Structured → lenient fallback only. Remove legacy path after verifying `inferred_constraints` extraction works with the new schema.

**Prerequisite:** PT.6 (legacy test consolidation) should be done first.

### 3.4 Externalize `MODEL_TOKEN_LIMITS` to config (PA.4)

**Problem:** Hardcoded dict requires code changes for new models. We just had to update it manually in PLAN1.

**Options:**
- Load from a TOML/JSON config file at startup
- Query provider capabilities dynamically
- Hybrid: config file with dynamic fallback

---

## Phase 4 — Validation & Sign-off

Final validation pass after all improvements are in place. Some items overlap with PLAN1's PM.4/PM.5 (which passed) but should be re-run after Phase 1–3 changes.

### 4.1 Full test suite (V.1)

Re-run `pytest tests/` after all phases complete. Confirm zero failures.

### 4.2 Contract tests (V.2)

Re-run `pytest tests/contract/` to verify envelope schemas remain valid after refactors.

### 4.3 Manual end-to-end test (V.3)

Run a deep research session with all features enabled:
- Fetch-time summarization ON
- Per-topic compression ON
- Forced reflection ON
- Clarification gate ON
- Role-based model routing ON

Verify the session completes successfully and produces meaningful output.

### 4.4 Token usage comparison (V.4)

Run a reference query before and after changes. Document token usage delta in the PR description.

### 4.5 Backward-compat session load (V.5)

Load a pre-existing saved research session (created before this branch). Confirm deserialization succeeds and the session can be resumed.

### 4.6 Config defaults review (V.6)

Audit all new config fields added across Phases 1–6 of the deep research work. Verify each has:
- A sensible default value
- Documentation (docstring or inline comment)
- Presence in any config schema/validation
