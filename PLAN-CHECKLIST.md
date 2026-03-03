# Alpha Branch Remediation Checklist

Track progress by replacing `[ ]` with `[x]` as items are completed.

---

## Phase 0 — Critical (Pre-Merge Blockers)

- [x] **0.1** Validate `report_output_path` before `write_text()` in `workflow_execution.py:409`
  - [x] Add `_validate_report_output_path()` helper with `.resolve()` and safe-directory check
  - [x] Reject `..` segments before resolution
  - [x] Add unit test for traversal attempt
- [x] **0.2** Wrap `response.json()` in `openalex.py:537` with `JSONDecodeError` handler
  - [x] Raise `SearchProviderError(retryable=False)`
  - [x] Add test for non-JSON response
- [x] **0.2** Wrap `response.json()` in `crossref.py:363` with `JSONDecodeError` handler
  - [x] Raise `SearchProviderError(retryable=False)`
  - [x] Add test for non-JSON response

---

## Phase 1 — High Priority (Fast-follow after merge)

- [x] **1.1** Use `CHARS_PER_TOKEN` constant in `claim_verification.py:325` instead of magic `3.5`
  - [x] Import from `_token_budget.py`
  - [x] Add system prompt overhead estimate (~500 tokens)
- [x] **1.2** Cap `_parse_extracted_claims()` at `max_claims * 2` in `claim_verification.py:200-269`
  - [x] Pass `max_claims` into parsing function
  - [x] Log warning when cap triggers
- [x] **1.3** Add `isinstance(pos, int) and pos >= 0` guard in `openalex.py:138-155`
  - [x] Log warning on invalid positions
- [x] **1.4** Add year bounds check `1000 <= year <= 2100` in `crossref.py:110`
- [x] **1.5** Decide threading fix strategy for `background_tasks.py` shared state
  - [x] Option A: `threading.Lock` around collection mutations (chosen)
  - [ ] ~~Option B: `queue.Queue` message passing~~ (not needed)
  - [x] Audit all mutation sites (~15 across 4-5 files)
  - [x] Implement chosen option
  - [x] Add thread-safe helpers to DeepResearchState

---

## Phase 2 — Medium Priority (Next sprint)

- [x] **2.1** Replace `except Exception: pass` with logging in `background_tasks.py:217-233`
- [x] **2.2** Track per-iteration source/finding IDs for rollback in `workflow_execution.py:545-572`
  - [x] Snapshot source/finding/topic_result counts at iteration entry via `_snapshot_iteration_counts()`
  - [x] Remove partial items on rollback via `_rollback_partial_iteration()`
  - [x] Store `rollback_counts` in metadata for resume validation
- [x] **2.3** Add per-entry size cap (100KB) to raw notes in `supervision.py:685-741`
  - [x] Added `MAX_RAW_NOTE_ENTRY_CHARS = 100_000` constant in `models/deep_research.py`
  - [x] Enforce truncation in `append_raw_note()` before aggregate caps
- [x] **2.4** Add decompression ratio limit to `docx_extractor.py`
  - [x] Check declared decompressed sizes from ZIP infolist
  - [x] Abort if ratio exceeds `MAX_DECOMPRESSION_RATIO` (100:1)
- [x] **2.5** Extract `check_gather_cancellation()` shared helper
  - [x] Created `_concurrency.py` module with `check_gather_cancellation()`
  - [x] Replaced manual checks in `supervision.py` (2 sites), `compression.py`, `claim_verification.py`

---

## Phase 3 — Low Priority (Backlog)

- [ ] **3.1** Consider stack-based HTML unescape in `_injection_protection.py:206`
- [ ] **3.2** Add metadata warning for JSON truncation repair in `claim_verification.py:232`
- [ ] **3.3** Escalate unknown OpenAlex filter keys to `ValueError` in `openalex.py:97`
- [ ] **3.4** Add API key constructor check in `semantic_scholar.py:207`
- [ ] **3.5** Add `MAX_EXTRACTED_CONTENT_LENGTH` check in `tavily.py:474`
- [ ] **3.6** Add comments to empty-list assertions in `test_supervision.py`
- [ ] **3.6** Consolidate scattered reflection tests
