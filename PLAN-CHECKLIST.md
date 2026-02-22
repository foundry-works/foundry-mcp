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

### 2b. Independent timeout per provider
- [x] Each provider (primary + fallbacks) gets the full configured timeout independently
- [x] Track wall-clock time via `method_start = time.monotonic()` for observability
- [x] Log actual wall-clock vs. configured timeout in result metadata
- [x] Fallback providers are tried after primary timeout with their own full timeout

### 2c. Tests for bounds and deadline
- [x] Create `tests/core/research/workflows/test_input_validation.py`
- [x] Test prompt at `MAX_PROMPT_LENGTH` → accepted
- [x] Test prompt over `MAX_PROMPT_LENGTH` → clear error
- [x] Test `max_iterations` at limit → accepted
- [x] Test `max_iterations` over limit → clear error
- [x] Test `max_sub_queries` over limit → clear error
- [x] Update `test_timeout_fallback.py` — each provider gets full timeout independently
- [x] Test primary timeout triggers fallback with full timeout
- [x] Test each provider receives the configured timeout value
- [x] Run: `pytest tests/core/research/ -v`

---

## Phase 3: Deep Research Thread Safety & Shutdown

### 3a. Cancellation event flag
- [x] Add `self._cancel_event = threading.Event()` to `DeepResearchWorkflow.__init__`
- [x] Check `_cancel_event.is_set()` at each phase boundary in `workflow_execution.py`
- [x] On cancel detection: persist current state, set status `CANCELLED`
- [x] Expose `cancel(session_id)` public method
- [x] Wire `cancel()` to deep-research `stop` action handler

### 3b. Graceful shutdown on SIGTERM
- [x] Register `signal.SIGTERM` handler in background task manager
- [x] Handler iterates active sessions, sets cancel events
- [x] Persist phase state before thread exit
- [x] Set session status to `INTERRUPTED` (distinct from `CANCELLED` and `FAILED`)
- [x] Log shutdown with session IDs and phase states
- [x] Ensure handler only affects deep research threads, not main MCP process

### 3c. Tests
- [x] Create `tests/core/research/workflows/test_deep_research_lifecycle.py`
- [x] Test cancel event stops execution between phases
- [x] Test cancel persists `CANCELLED` status with correct phase state
- [x] Test SIGTERM handler sets cancel events on all active sessions
- [x] Test `INTERRUPTED` status is distinguishable from `CANCELLED` and `FAILED`
- [x] Test cancel on already-completed session is a no-op
- [x] Run: `pytest tests/core/research/ -v`

---

## Phase 4: Provider Boilerplate Consolidation

### 4a. Extract shared utilities
- [x] Consolidate `parse_retry_after()` (5 copies → 1 in `shared.py`)
- [x] Update all 5 provider imports for `parse_retry_after`
- [x] Consolidate `extract_domain()` (3 copies → 1 in `shared.py`)
- [x] Update all 3 provider imports for `extract_domain`
- [x] Consolidate `parse_iso_date()` (5 copies → 1 in `shared.py`)
- [x] Update all 5 provider imports for `parse_iso_date`
- [x] Consolidate validation error classification (4 copies → `classify_http_error()` factory)
- [x] Update all 4 provider imports for classification
- [x] Delete all inline copies after import verification
- [x] Remove dead `_parse_retry_after()` wrappers from all 5 providers
- [x] Remove dead `_parse_date()` wrappers from google, perplexity (keep semantic_scholar — adds `extra_formats`)
- [x] Remove dead `_extract_domain()` wrapper from perplexity; inline in tavily_extract
- [x] Update `test_provider_characterization.py` — remove `_parse_retry_after` references
- [x] Update `test_perplexity.py` — use shared functions instead of removed wrappers

### 4b. Provider-specific error classifier registry
- [x] Add `ERROR_CLASSIFIERS: ClassVar[dict[int, ErrorType]]` class variable to `SearchProvider`
- [x] Add `extract_status_code()` helper to `shared.py`
- [x] Add `_ERROR_TYPE_DEFAULTS` mapping to `shared.py`
- [x] Update default `classify_error()` to check registry then delegate to `classify_http_error`
- [x] Google: register 403-quota, 429-rate-limit classifiers (keeps custom `classify_error` override for `RateLimitError` quota detection)
- [x] Perplexity: register 429-rate-limit classifier, remove `classify_error` override
- [x] SemanticScholar: register 504-gateway classifier, remove `classify_error` override
- [x] Tavily: remove `classify_error` override (uses base default, empty registry)
- [x] TavilyExtract: keep `classify_error` (standalone class, not a `SearchProvider` subclass)
- [x] Verify Tavily + TavilyExtract work with defaults (no custom classifiers needed)

### 4c. Tests
- [x] Create `tests/core/research/providers/test_shared_utils.py`
- [x] Test `parse_retry_after()`: valid header, missing header, malformed header, zero, large, RFC date
- [x] Test `extract_domain()`: standard URL, subdomain, unicode, IP address, malformed
- [x] Test `parse_iso_date()`: ISO format, timezone-aware, malformed, None, year-only with extra_formats
- [x] Test `extract_status_code()`: standard format, API error, bare code, no code, edge cases
- [x] Test `ERROR_CLASSIFIERS` registry: all providers have expected entries
- [x] Test registry matches status codes in `SearchProviderError` messages
- [x] Test registry falls through for unregistered codes
- [x] Test providers without custom classifiers use defaults correctly
- [x] Test all providers classify auth/timeout errors consistently
- [x] Run: `pytest tests/core/research/providers/ -v` (579 passed)
- [x] Fix ruff lint issues (unused imports in `base.py`, `ErrorClassification` annotation in `tavily_extract.py`, import sorting in test)
- [x] Run: `pytest tests/core/research/ -v` (1381 passed, 6 skipped)

---

## Final
- [ ] Run: `pytest tests/ --timeout=120`
- [ ] Run: `ruff check src/foundry_mcp/core/research/`
- [ ] Verify no silent data loss in any workflow (parse failures always logged)
- [ ] Update CHANGELOG.md
- [ ] Commit, push, PR to `beta`
