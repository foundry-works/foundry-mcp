# Research Tools Hardening — Checklist

## Phase 0: Test Safety Net

### 0a. Provider error classification tests
- [ ] Create `tests/core/research/providers/test_error_classification.py`
- [ ] Test Google `classify_error()` — 403 quota vs. 403 forbidden
- [ ] Test Google `classify_error()` — 429 rate limit with Retry-After header
- [ ] Test Perplexity default classification — 429 rate limit
- [ ] Test SemanticScholar default classification — 504 gateway timeout
- [ ] Test `base.py` fallback for unrecognized status codes
- [ ] Test circuit breaker transitions match classification output

### 0b. Timeout + fallback behavior tests
- [ ] Create `tests/core/research/workflows/test_timeout_fallback.py`
- [ ] Test primary provider timeout triggers fallback
- [ ] Test total wall-clock with primary + fallback timeout (document current bug)
- [ ] Test no fallback configured — timeout returns error cleanly

### 0c. Parse failure edge case tests
- [ ] Create `tests/core/research/workflows/test_parse_edge_cases.py`
- [ ] ThinkDeep: empty LLM response → no crash, empty hypotheses
- [ ] ThinkDeep: JSON-formatted response → currently not parsed (baseline)
- [ ] ThinkDeep: false positive keywords ("unsupported") → document behavior
- [ ] Ideate `_parse_ideas()`: numbered list format
- [ ] Ideate `_parse_ideas()`: markdown header format
- [ ] Ideate `_parse_clusters()`: inconsistent spacing
- [ ] Ideate `_parse_scores()`: multi-digit scores, decimal scores
- [ ] DeepResearch `_parse_analysis_response()`: truncated output
- [ ] DeepResearch `_parse_analysis_response()`: alternative heading styles

### 0d. Consensus "majority" strategy test
- [ ] Create `tests/core/research/workflows/test_consensus_majority.py`
- [ ] Test 3 providers, 2 agree, 1 diverges → majority wins
- [ ] Test 3 providers, all different → tie-breaking behavior
- [ ] Test fewer responses than `min_responses` → appropriate error
- [ ] Run: `pytest tests/core/research/ -v`

---

## Phase 1: Structured Output for Response Parsing

### 1a. ThinkDeep structured output
- [x] Define `HypothesisUpdate` Pydantic model with `hypothesis`, `evidence: list[EvidenceItem]`
- [x] Define `EvidenceItem` with `text`, `strength: Literal["strong", "moderate", "weak"]`, `source`
- [x] Update ThinkDeep system prompt to request JSON conforming to schema
- [x] Replace keyword matching in `_update_hypotheses_from_response()` (lines 251–301) with JSON parse
- [x] Add Pydantic validation with clear error messages
- [x] Add fallback: JSON parse failure → current keyword extraction + warning log
- [x] Update confidence scoring to use `evidence.strength` instead of fixed-step bumping
- [x] Add `parse_method` to result metadata ("json" | "fallback_keyword")

### 1b. Ideate structured output
- [x] Define `IdeaOutput` Pydantic model
- [x] Define `ClusterOutput` Pydantic model
- [x] Define `ScoreOutput` Pydantic model
- [x] Update divergent phase system prompt → JSON
- [x] Update clustering phase system prompt → JSON
- [x] Update scoring phase system prompt → JSON
- [x] Replace `_parse_ideas()` (lines 512–545) with JSON parse + validation
- [x] Replace `_parse_clusters()` (lines 546–594) with JSON parse + validation
- [x] Replace `_parse_scores()` (lines 595–616) with JSON parse + validation
- [x] Add fallback: JSON failure → current regex + warning log
- [x] Add parse success/failure count to workflow result metadata

### 1c. Deep Research analysis structured output
- [x] Define `AnalysisFinding` Pydantic model with `claim`, `evidence`, `confidence`, `source_urls`
- [x] Update analysis prompt in `_analysis_prompts.py` to request JSON array
- [x] Replace markdown parsing in `_analysis_parsing.py` with JSON parse + validation
- [x] Add fallback to current markdown parsing + warning log
- [x] Surface `parse_method` in phase metadata

### 1d. Update tests for structured output
- [x] Update `test_parse_edge_cases.py` — valid JSON parsed correctly (ThinkDeep)
- [x] Update `test_parse_edge_cases.py` — valid JSON parsed correctly (Ideate)
- [x] Update `test_parse_edge_cases.py` — valid JSON parsed correctly (DeepResearch)
- [x] Test malformed JSON triggers fallback + logs warning
- [x] Test Pydantic validation catches missing required fields
- [x] Test `parse_method` metadata correctly set in all workflows
- [x] Run: `pytest tests/core/research/ -v`

---

## Phase 2: Input Validation & Timeout Fix

### 2a. Input bounds validation
- [x] Define validation constants (`MAX_PROMPT_LENGTH`, `MAX_ITERATIONS`, `MAX_SUB_QUERIES`, `MAX_SOURCES_PER_QUERY`, `MAX_CONCURRENT_PROVIDERS`)
- [x] Add prompt length validation in base workflow entry point
- [x] Add `max_iterations` validation in deep research `start()` handler
- [x] Add `max_sub_queries` validation in deep research `start()` handler
- [x] Add `max_sources_per_query` validation in deep research `start()` handler
- [x] Return clear error envelope on bound violation (not exception)

### 2b. Deadline-based timeout
- [x] Compute `deadline = time.monotonic() + timeout_seconds` at `_execute_provider_async` entry
- [x] Pass `remaining = max(0, deadline - time.monotonic())` to primary provider attempt
- [x] Pass remaining budget to each fallback attempt
- [x] Skip fallback if `remaining <= 0`, return timeout error with elapsed duration
- [x] Log actual wall-clock vs. configured timeout in result metadata
- [x] Update all callers that pass timeout values

### 2c. Tests for bounds and deadline
- [x] Create `tests/core/research/workflows/test_input_validation.py`
- [x] Test prompt at `MAX_PROMPT_LENGTH` → accepted
- [x] Test prompt over `MAX_PROMPT_LENGTH` → clear error
- [x] Test `max_iterations` at limit → accepted
- [x] Test `max_iterations` over limit → clear error
- [x] Test `max_sub_queries` over limit → clear error
- [x] Update `test_timeout_fallback.py` — deadline caps total duration (was failing, now passes)
- [x] Test deadline: primary consumes 280s of 300s budget → fallback gets 20s
- [x] Test deadline: primary consumes full budget → fallback skipped
- [x] Run: `pytest tests/core/research/ -v`

---

## Phase 3: Deep Research Thread Safety & Shutdown

### 3a. Cancellation event flag
- [ ] Add `self._cancel_event = threading.Event()` to `DeepResearchWorkflow.__init__`
- [ ] Check `_cancel_event.is_set()` at each phase boundary in `workflow_execution.py`
- [ ] On cancel detection: persist current state, set status `CANCELLED`
- [ ] Expose `cancel(session_id)` public method
- [ ] Wire `cancel()` to deep-research `stop` action handler

### 3b. Graceful shutdown on SIGTERM
- [ ] Register `signal.SIGTERM` handler in background task manager
- [ ] Handler iterates active sessions, sets cancel events
- [ ] Persist phase state before thread exit
- [ ] Set session status to `INTERRUPTED` (distinct from `CANCELLED` and `FAILED`)
- [ ] Log shutdown with session IDs and phase states
- [ ] Ensure handler only affects deep research threads, not main MCP process

### 3c. Tests
- [ ] Create `tests/core/research/workflows/test_deep_research_lifecycle.py`
- [ ] Test cancel event stops execution between phases
- [ ] Test cancel persists `CANCELLED` status with correct phase state
- [ ] Test SIGTERM handler sets cancel events on all active sessions
- [ ] Test `INTERRUPTED` status is distinguishable from `CANCELLED` and `FAILED`
- [ ] Test cancel on already-completed session is a no-op
- [ ] Run: `pytest tests/core/research/ -v`

---

## Phase 4: Provider Boilerplate Consolidation

### 4a. Extract shared utilities
- [ ] Consolidate `parse_retry_after()` (5 copies → 1 in `shared.py`)
- [ ] Update all 5 provider imports for `parse_retry_after`
- [ ] Consolidate `extract_domain()` (3 copies → 1 in `shared.py`)
- [ ] Update all 3 provider imports for `extract_domain`
- [ ] Consolidate `parse_iso_date()` (5 copies → 1 in `shared.py`)
- [ ] Update all 5 provider imports for `parse_iso_date`
- [ ] Consolidate validation error classification (4 copies → `classify_http_error()` factory)
- [ ] Update all 4 provider imports for classification
- [ ] Delete all inline copies after import verification

### 4b. Provider-specific error classifier registry
- [ ] Add `ERROR_CLASSIFIERS: dict[int, ErrorType]` class variable to `SearchProvider`
- [ ] Update default `classify_error()` to check registry before generic fallback
- [ ] Google: register 403-quota, 429-rate-limit classifiers
- [ ] Perplexity: register 429-rate-limit classifier
- [ ] SemanticScholar: register 504-gateway classifier
- [ ] Verify Tavily + TavilyExtract work with defaults (no custom classifiers needed)

### 4c. Tests
- [ ] Create `tests/core/research/providers/test_shared_utils.py`
- [ ] Test `parse_retry_after()`: valid header, missing header, malformed header
- [ ] Test `extract_domain()`: standard URL, subdomain, unicode, malformed
- [ ] Test `parse_iso_date()`: ISO format, timezone-aware, malformed, None
- [ ] Test `classify_http_error()`: known codes, unknown codes, edge cases
- [ ] Update `test_error_classification.py` — registry-based classifiers work
- [ ] Test providers without custom classifiers use defaults correctly
- [ ] Run: `pytest tests/core/research/ -v`

---

## Final
- [ ] Run: `pytest tests/ --timeout=120`
- [ ] Run: `ruff check src/foundry_mcp/core/research/`
- [ ] Verify no silent data loss in any workflow (parse failures always logged)
- [ ] Update CHANGELOG.md
- [ ] Commit, push, PR to `beta`
