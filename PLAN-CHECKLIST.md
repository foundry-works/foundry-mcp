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

- [ ] **1.1** Use `CHARS_PER_TOKEN` constant in `claim_verification.py:325` instead of magic `3.5`
  - [ ] Import from `_token_budget.py`
  - [ ] Add system prompt overhead estimate (~500 tokens)
- [ ] **1.2** Cap `_parse_extracted_claims()` at `max_claims * 2` in `claim_verification.py:200-269`
  - [ ] Pass `max_claims` into parsing function
  - [ ] Log warning when cap triggers
- [ ] **1.3** Add `isinstance(pos, int) and pos >= 0` guard in `openalex.py:138-155`
  - [ ] Log warning on invalid positions
- [ ] **1.4** Add year bounds check `1000 <= year <= 2100` in `crossref.py:110`
- [ ] **1.5** Decide threading fix strategy for `background_tasks.py` shared state
  - [ ] Option A: `threading.Lock` around collection mutations
  - [ ] Option B: `queue.Queue` message passing
  - [ ] Audit all mutation sites (~15 across 4-5 files)
  - [ ] Implement chosen option
  - [ ] Add concurrency test

---

## Phase 2 — Medium Priority (Next sprint)

- [ ] **2.1** Replace `except Exception: pass` with logging in `background_tasks.py:217-233`
- [ ] **2.2** Track per-iteration source/finding IDs for rollback in `workflow_execution.py:545-572`
  - [ ] Add `iteration_N_source_ids` to state metadata
  - [ ] Remove partial items on rollback
  - [ ] Validate counts on resume
- [ ] **2.3** Add per-entry size cap (100KB) to raw notes in `supervision.py:685-741`
- [ ] **2.4** Add decompression ratio limit to `docx_extractor.py`
  - [ ] Track cumulative decompressed bytes
  - [ ] Abort if ratio exceeds 100:1
- [ ] **2.5** Extract `check_gather_cancellation()` shared helper
  - [ ] Create helper in `_helpers.py` or similar
  - [ ] Replace manual checks in `supervision.py`, `compression.py`, `claim_verification.py`, `citation_network.py`

---

## Phase 3 — Low Priority (Backlog)

- [ ] **3.1** Consider stack-based HTML unescape in `_injection_protection.py:206`
- [ ] **3.2** Add metadata warning for JSON truncation repair in `claim_verification.py:232`
- [ ] **3.3** Escalate unknown OpenAlex filter keys to `ValueError` in `openalex.py:97`
- [ ] **3.4** Add API key constructor check in `semantic_scholar.py:207`
- [ ] **3.5** Add `MAX_EXTRACTED_CONTENT_LENGTH` check in `tavily.py:474`
- [ ] **3.6** Add comments to empty-list assertions in `test_supervision.py`
- [ ] **3.6** Consolidate scattered reflection tests
