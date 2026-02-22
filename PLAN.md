# Research Tools Hardening — v0.15

## Context

Audit of the research subsystem (~15,000 LOC, 88 files, 5 workflows, 5 search providers) identified systemic issues affecting research quality, safety, and maintainability. Multi-model consensus confirmed the priority ordering below.

The research subsystem powers chat, consensus, thinkdeep, ideate, and deep-research workflows. The core issues are: silent data loss from fragile LLM response parsing, missing input guards, a timeout bug that can double wall-clock wait times, and insufficient test coverage for failure modes.

---

## Phase 0: Test Safety Net

**Goal:** Establish test coverage for unhappy paths before changing any production code. This phase is the force multiplier that makes all subsequent phases safe to ship.

### 0a. Provider error classification tests

**Files:**
- `tests/core/research/providers/test_error_classification.py` (new)

**Changes:**
- Test each provider's `classify_error()` override against known error codes (Google 403 quota vs. 403 forbidden, Perplexity 429 rate limit, SemanticScholar 504 gateway)
- Test default fallback in `base.py:205+` for unrecognized status codes
- Test that circuit breaker state transitions match classification output

### 0b. Timeout + fallback behavior tests

**Files:**
- `tests/core/research/workflows/test_timeout_fallback.py` (new)

**Changes:**
- Test that primary provider timeout triggers fallback
- Test total wall-clock time when primary + fallback both hit timeout (**documents current bug**)
- Test that deadline-based timeout caps total duration (will fail until Phase 2 fix)

### 0c. Parse failure edge case tests

**Files:**
- `tests/core/research/workflows/test_parse_edge_cases.py` (new)

**Changes:**
- ThinkDeep: test `_update_hypotheses_from_response()` with empty response, JSON response, response missing keywords, response with false-positive keywords ("unsupported" containing "support")
- Ideate: test `_parse_ideas()` / `_parse_clusters()` / `_parse_scores()` with varied LLM formatting (numbered lists, markdown headers, inconsistent spacing, JSON output)
- Deep Research: test `_parse_analysis_response()` with truncated output, missing sections, alternative heading styles

### 0d. Consensus "majority" strategy test

**Files:**
- `tests/core/research/workflows/test_consensus_majority.py` (new)

**Changes:**
- Test majority strategy with 3 providers (2 agree, 1 diverges)
- Test tie-breaking behavior
- Test with fewer responses than `min_responses`

---

## Phase 1: Structured Output for Response Parsing

**Goal:** Replace fragile regex/keyword parsing with JSON-schema-constrained LLM output. This is the highest-impact change — eliminates silent data loss across three workflows.

### 1a. ThinkDeep structured output

**Files:**
- `src/foundry_mcp/core/research/workflows/thinkdeep.py` (lines 251–301)

**Changes:**
- Define Pydantic models: `HypothesisUpdate`, `EvidenceItem` with explicit `strength: Literal["strong", "moderate", "weak"]`
- Update system prompt to request JSON output conforming to schema
- Replace `_update_hypotheses_from_response()` keyword matching with JSON parse + Pydantic validation
- Add fallback: if JSON parse fails, attempt current keyword extraction, log warning with `parse_method: "fallback_keyword"`
- Update confidence scoring to use evidence `strength` field instead of fixed-step bumping

### 1b. Ideate structured output

**Files:**
- `src/foundry_mcp/core/research/workflows/ideate.py` (lines 512–616)

**Changes:**
- Define Pydantic models: `IdeaOutput`, `ClusterOutput`, `ScoreOutput`
- Update system prompts for each phase (divergent, clustering, scoring) to request JSON
- Replace `_parse_ideas()`, `_parse_clusters()`, `_parse_scores()` with JSON parse + validation
- Add fallback: attempt current regex parsing on JSON failure, log warning
- Add parse success/failure metrics to workflow result metadata

### 1c. Deep Research analysis structured output

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_analysis_parsing.py` (131 lines)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_analysis_prompts.py` (187 lines)

**Changes:**
- Define Pydantic model: `AnalysisFinding` with `claim`, `evidence`, `confidence`, `source_urls` fields
- Update analysis prompt to request JSON array of findings
- Replace format-specific parsing in `_parse_analysis_response()` with JSON parse + validation
- Add fallback to current markdown parsing, log warning
- Surface parse method in phase metadata for observability

### 1d. Update Phase 0 tests to validate new behavior

**Files:**
- `tests/core/research/workflows/test_parse_edge_cases.py` (update)

**Changes:**
- Add tests: valid JSON parsed correctly for each workflow
- Add tests: malformed JSON triggers fallback, logs warning
- Add tests: Pydantic validation catches missing required fields
- Add tests: parse method metadata correctly set ("json" vs "fallback_keyword" vs "fallback_regex")

---

## Phase 2: Input Validation & Timeout Fix

**Goal:** Prevent runaway resource consumption and fix the timeout contract breach.

### 2a. Input bounds validation

**Files:**
- `src/foundry_mcp/core/research/workflows/base.py` (top of `execute()` or equivalent entry points)
- `src/foundry_mcp/core/research/workflows/deep_research/core.py`

**Changes:**
- Add validation constants to a `_constants.py` or inline:
  - `MAX_PROMPT_LENGTH = 50_000` (chars)
  - `MAX_ITERATIONS = 10`
  - `MAX_SUB_QUERIES = 20`
  - `MAX_SOURCES_PER_QUERY = 50`
  - `MAX_CONCURRENT_PROVIDERS = 10`
- Validate at workflow entry points, return clear error envelope on violation
- Deep research: validate `max_iterations`, `max_sub_queries`, `max_sources_per_query` in `start()` action handler
- All workflows: validate prompt length in base `_execute_provider_async()`

### 2b. Deadline-based timeout

**Files:**
- `src/foundry_mcp/core/research/workflows/base.py` (lines 286–528, `_execute_provider_async`)

**Changes:**
- Compute `deadline = time.monotonic() + timeout_seconds` at method entry
- Pass `remaining = max(0, deadline - time.monotonic())` to each provider attempt (primary + fallbacks)
- If `remaining <= 0` before fallback attempt, skip it and return timeout error
- Log actual wall-clock duration vs. configured timeout in result metadata
- Update all callers that pass timeout values

### 2c. Tests for bounds and deadline

**Files:**
- `tests/core/research/workflows/test_input_validation.py` (new)
- `tests/core/research/workflows/test_timeout_fallback.py` (update)

**Changes:**
- Test each bound: at limit (pass), over limit (clear error)
- Test deadline: primary timeout consumes budget, fallback gets remainder
- Test deadline: no time left = fallback skipped with appropriate message
- Update Phase 0 timeout test to expect new deadline behavior (was documenting bug, now should pass)

---

## Phase 3: Deep Research Thread Safety & Shutdown

**Goal:** Prevent orphaned research sessions and ensure clean process termination.

### 3a. Cancellation event flag

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py` (272 lines)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (519 lines)

**Changes:**
- Add `self._cancel_event = threading.Event()` in `__init__`
- Check `_cancel_event.is_set()` at phase boundaries in `workflow_execution.py`
- Expose `cancel(session_id)` method that sets the event and persists `CANCELLED` status
- Wire cancel to deep-research `stop` action handler

### 3b. Graceful shutdown on SIGTERM

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py`

**Changes:**
- Register `signal.SIGTERM` handler that iterates active sessions and sets cancel events
- On cancel: persist current phase state before exiting thread
- Set session status to `INTERRUPTED` (distinguishable from `CANCELLED` and `FAILED`)
- Log shutdown with session IDs and phase states

### 3c. Tests

**Files:**
- `tests/core/research/workflows/test_deep_research_lifecycle.py` (new)

**Changes:**
- Test cancel event stops execution between phases
- Test cancel persists CANCELLED status
- Test SIGTERM handler sets cancel events on all active sessions
- Test INTERRUPTED status distinguishable from CANCELLED and FAILED

---

## Phase 4: Provider Boilerplate Consolidation

**Goal:** Reduce ~250 lines of duplicated utility code across 5 providers.

### 4a. Extract shared utilities

**Files:**
- `src/foundry_mcp/core/research/providers/shared.py` (626 lines, extend)

**Changes:**
- Consolidate duplicate `parse_retry_after()` (5 copies) into single shared implementation
- Consolidate duplicate `extract_domain()` (3 copies) into shared
- Consolidate duplicate `parse_iso_date()` (5 copies) into shared
- Consolidate duplicate validation error classification (4 copies) into shared `classify_http_error()` factory
- Each consolidation: update all provider imports, delete inline copies

### 4b. Provider-specific error classifier registry

**Files:**
- `src/foundry_mcp/core/research/providers/base.py` (lines 205+)
- `src/foundry_mcp/core/research/providers/google.py`
- `src/foundry_mcp/core/research/providers/perplexity.py`
- `src/foundry_mcp/core/research/providers/semantic_scholar.py`

**Changes:**
- Add `ERROR_CLASSIFIERS: dict[int, ErrorType]` class variable pattern to `SearchProvider`
- Google: register 403-quota, 429-rate-limit classifiers
- Perplexity: register 429-rate-limit classifier (currently using defaults)
- SemanticScholar: register 504-gateway classifier (currently using defaults)
- Default `classify_error()` checks registry before falling back to generic classification

### 4c. Tests

**Files:**
- `tests/core/research/providers/test_shared_utils.py` (new)
- `tests/core/research/providers/test_error_classification.py` (update from Phase 0)

**Changes:**
- Test each shared utility with edge cases (malformed dates, missing headers, unicode domains)
- Update Phase 0 error classification tests to cover new registry pattern
- Test that providers without custom classifiers still work via defaults

---

## Implementation Sequence

```
Phase 0 (tests)       ──── no deps, do first
    │
Phase 1 (parsing)     ──── depends on Phase 0 for safety
    │
Phase 2 (validation)  ──── depends on Phase 0; independent of Phase 1
    │
Phase 3 (lifecycle)   ──── depends on Phase 0; independent of Phases 1-2
    │
Phase 4 (providers)   ──── depends on Phase 0; independent of Phases 1-3
```

Phases 1–4 are independent of each other and can be parallelized after Phase 0.

---

## Key Files

| File | Phases | Lines |
|------|--------|-------|
| `core/research/workflows/thinkdeep.py` | 1a | 412 |
| `core/research/workflows/ideate.py` | 1b | 695 |
| `core/research/workflows/deep_research/phases/_analysis_parsing.py` | 1c | 131 |
| `core/research/workflows/deep_research/phases/_analysis_prompts.py` | 1c | 187 |
| `core/research/workflows/base.py` | 2a, 2b | 546 |
| `core/research/workflows/deep_research/core.py` | 2a | 254 |
| `core/research/workflows/deep_research/background_tasks.py` | 3a, 3b | 272 |
| `core/research/workflows/deep_research/workflow_execution.py` | 3a | 519 |
| `core/research/providers/shared.py` | 4a | 626 |
| `core/research/providers/base.py` | 4b | 274 |
| `core/research/providers/google.py` | 4b | 492 |
| `core/research/providers/perplexity.py` | 4b | 519 |
| `core/research/providers/semantic_scholar.py` | 4b | 579 |

All paths relative to `src/foundry_mcp/`.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Structured output degrades LLM response quality | Fallback to current parsing on JSON failure; monitor parse method metrics |
| Input bounds too restrictive for legitimate use | Set generous defaults; make configurable via config |
| Timeout deadline change breaks existing integrations | Phase 0 tests document current behavior first; deadline is strictly more correct |
| Provider consolidation introduces regressions | Phase 0 error classification tests run before and after consolidation |
| SIGTERM handler conflicts with MCP server shutdown | Register handler only for deep research daemon threads, not main process |
