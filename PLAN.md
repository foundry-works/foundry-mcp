# Alpha Branch Remediation Plan

**Branch:** `alpha` (44 commits, 186 files, +68,826 / -8,970 lines)
**Review date:** 2026-03-03
**Verdict:** APPROVE with follow-up items

---

## Phase 0 — Critical Fixes (Pre-Merge Blockers)

### 0.1 Path Traversal in Report Output

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py:409-410`

**Problem:** `state.report_output_path` is written to via `Path(...).write_text()` with no validation or normalization. User-controlled input could write to arbitrary filesystem locations (e.g., `../../etc/cron.d/backdoor`).

**Fix:**
- Add a `_validate_report_output_path(path, safe_basedir)` helper
- Call `Path(path).resolve()` and enforce the resolved path is within the research output directory
- Reject paths containing `..` segments before resolution as a belt-and-suspenders check
- Apply validation before the `write_text()` call

**Scope:** ~15 lines, 1 file + 1 test file

---

### 0.2 Missing JSON Parse Exception Handling in API Providers

**Files:**
- `src/foundry_mcp/core/research/providers/openalex.py:537`
- `src/foundry_mcp/core/research/providers/crossref.py:363`

**Problem:** `response.json()` calls lack `try/except json.JSONDecodeError`. A proxy intercept, API misconfiguration, or rate-limit HTML page causes an unhandled crash.

**Fix:**
- Wrap each `response.json()` with:
  ```python
  try:
      return response.json()
  except json.JSONDecodeError as e:
      raise SearchProviderError(
          provider="<name>",
          message="Invalid JSON in API response",
          retryable=False,
          original_error=e,
      ) from e
  ```

**Scope:** ~10 lines per provider, 2 files + 2 test files

---

## Phase 1 — High Priority (Merge, then fast-follow)

### 1.1 Token Estimation Constant Inconsistency

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py:325`
- `src/foundry_mcp/core/research/workflows/deep_research/_token_budget.py:12`

**Problem:** Claim verification uses `len(text) / 3.5` for token estimation but the canonical constant is `CHARS_PER_TOKEN = 4`. The ~12.5% over-estimation can cause unnecessary claim drops or, if flipped, over-budget LLM calls.

**Fix:**
- Import and use `CHARS_PER_TOKEN` from `_token_budget.py` in claim verification
- Add ~500 token overhead estimate for system prompt and JSON structure
- Validate per-claim budget doesn't exceed `max_input_tokens`

**Scope:** ~10 lines, 1 file

---

### 1.2 Unbounded Claim Extraction from LLM

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py:200-269`

**Problem:** `_parse_extracted_claims()` has no cardinality limit. A hallucinating LLM returning 10,000 claims causes memory pressure before the `max_claims` filter at line 780 applies.

**Fix:**
- Add early cap in `_parse_extracted_claims()`: stop appending after `max_claims * 2` entries
- Pass `max_claims` config value into the parsing function
- Log a warning when cap is hit

**Scope:** ~8 lines, 1 file

---

### 1.3 Type-Unvalidated Position Values in OpenAlex Abstract Reconstruction

**File:** `src/foundry_mcp/core/research/providers/openalex.py:138-155`

**Problem:** Position values from the inverted index API response are not type-checked. Non-integer, negative, or extremely large positions pass through silently.

**Fix:**
- Add `isinstance(pos, int) and pos >= 0` guard in the loop
- Log warning and `continue` on invalid positions

**Scope:** ~5 lines, 1 file

---

### 1.4 Crossref Year Parsing Without Bounds

**File:** `src/foundry_mcp/core/research/providers/crossref.py:110`

**Problem:** `int(parts[0][0])` has no bounds validation. Values like year 99999 or negative years are accepted.

**Fix:**
- Add bounds check: `year = int(parts[0][0]); return year if 1000 <= year <= 2100 else None`

**Scope:** ~3 lines, 1 file

---

### 1.5 Shared Mutable State Race Condition

**File:** `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py:99-106`

**Problem:** `DeepResearchState` is shared between daemon and main threads. Collection mutations (`sources.extend()`, `findings.extend()`, `raw_notes.append()`) are not atomic under the GIL for composite read-modify-write.

**Fix — Option A (Lock):**
- Add a `threading.Lock` to state or background task context
- Wrap all collection mutations in `with state_lock:`
- Audit all mutation sites across supervision, gathering, and synthesis phases

**Fix — Option B (Message Queue):**
- Daemon thread appends results to a `queue.Queue`
- Main thread drains queue into state at safe sync points

**Scope:** Medium — requires design decision and audit of ~15 mutation sites across 4-5 files

---

## Phase 2 — Medium Priority (Next sprint)

### 2.1 Bare Exception Handlers in Cleanup

**File:** `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py:217-233`

**Problem:** Three `except Exception: pass` blocks around audit, state flush, and task registry cleanup. Failures are never logged, making debugging impossible.

**Fix:** Replace `pass` with `logger.debug("Cleanup failed: %s", e, exc_info=True)`

---

### 2.2 Incomplete Cancellation Rollback

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py:545-572`

**Problem:** On partial-iteration rollback, sources/findings from the partial iteration are retained. Resume logic doesn't validate consistency with the rolled-back iteration number.

**Fix:**
- Track items added per iteration (e.g., `state.metadata["iteration_N_source_ids"]`)
- On rollback, remove items from the current iteration
- On resume, validate source/finding counts match expected iteration state

---

### 2.3 Raw Notes Per-Entry Size Cap

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py:685-741`

**Problem:** Count cap (50) and total char cap (500K) exist, but a single entry can be arbitrarily large before truncation.

**Fix:** Add per-entry cap (e.g., 100KB) at append time, before aggregate truncation.

---

### 2.4 Zip Decompression Bomb Protection

**File:** `src/foundry_mcp/core/research/docx_extractor.py`

**Problem:** Compressed file size is gated at 10MB, but no decompression ratio limit. A crafted DOCX could decompress to gigabytes.

**Fix:** Track cumulative decompressed bytes during extraction; abort if ratio exceeds threshold (e.g., 100:1).

---

### 2.5 Extract `asyncio.gather` CancelledError Propagation Helper

**Files:** `supervision.py:1317`, `compression.py:997`, `claim_verification.py:496`, `citation_network.py:174`

**Problem:** Manual `CancelledError` check after `gather(return_exceptions=True)` is repeated in 4+ places inconsistently.

**Fix:** Create shared helper:
```python
def check_gather_cancellation(results: list) -> None:
    for r in results:
        if isinstance(r, (asyncio.CancelledError, KeyboardInterrupt)):
            raise r
```

---

## Phase 3 — Low Priority (Backlog)

### 3.1 HTML Unescape Loop Depth
`_injection_protection.py:206` — 5-round cap may leave deeply encoded payloads. Consider stack-based exhaustive decode.

### 3.2 JSON Truncation Repair
`claim_verification.py:232` — Blindly appending `]` to truncated JSON can create semantically wrong structures. Add metadata warning.

### 3.3 OpenAlex Silent Filter Key Drop
`openalex.py:97-113` — Unknown filter keys log a warning but are silently dropped. Consider raising `ValueError`.

### 3.4 Semantic Scholar Constructor Key Validation
`semantic_scholar.py:207` — API key not validated at construction time; fails only at runtime.

### 3.5 DOCX Extraction Size Limit After Decompression
`tavily.py:474-500` — Extracted DOCX text returned without size validation. Add `MAX_EXTRACTED_CONTENT_LENGTH` check.

### 3.6 Test Housekeeping
- Add clarifying comments to empty-list assertions in `test_supervision.py` (lines 799, 2353, 2537)
- Consolidate scattered reflection tests into a dedicated section
