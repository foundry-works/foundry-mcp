# Implementation Checklist: Fix Citation Ordering After Claim Verification

## Phase 1: Split `postprocess_citations` into two stages

- [ ] **1.1** In `_citation_postprocess.py`, create `cleanup_citations(report, state, *, query_type=None) -> tuple[str, dict]`
  - Steps 1-3: extract cited numbers, strip LLM sources section, remove dangling citations
  - Return `(cleaned_report, cleanup_metadata)` with keys: `total_citations_in_report`, `total_sources_with_numbers`, `dangling_citations_removed`
- [ ] **1.2** In `_citation_postprocess.py`, create `finalize_citations(report, state, *, query_type=None) -> tuple[str, dict]`
  - Steps 4-5: renumber citations to reading order, append deterministic bibliography
  - Return `(finalized_report, finalize_metadata)` with keys: `renumbered_count`, `unreferenced_sources`, `format_style`, `total_citations_in_report`
- [ ] **1.3** Refactor `postprocess_citations` to call `cleanup_citations` then `finalize_citations`, merging metadata dicts
  - Must produce identical output to current implementation (no behavioral change for callers)
- [ ] **1.4** Add `needs_renumber(report, *, max_citation=None) -> bool` helper
  - Scan first-appearance order of `[N]` citations; return True if not sequential from 1
- [ ] **1.5** Export new functions in module `__all__` or ensure they are importable

## Phase 2: Rewire synthesis to use cleanup-only stage

- [x] **2.1** In `synthesis.py` `_build_result()`, replace `postprocess_citations` import with `cleanup_citations`
- [x] **2.2** Call `cleanup_citations(report, state, query_type=query_type)` instead of `postprocess_citations`
- [x] **2.3** Store `query_type` in `state.metadata["_query_type"]` for downstream finalize
- [x] **2.4** Update `citation_metadata` variable usage — cleanup returns a subset of original keys; adjust audit log fields if needed

## Phase 3: Add finalize step after claim verification

- [ ] **3.1** In `workflow_execution.py`, import `finalize_citations` from `_citation_postprocess`
- [ ] **3.2** After the claim verification block (after line ~539, before `mark_completed`), add finalize call:
  - `report, finalize_meta = finalize_citations(state.report, state, query_type=state.metadata.get("_query_type"))`
  - `state.report = report`
- [ ] **3.3** Handle the case where claim verification is disabled/skipped — finalize must still run
- [ ] **3.4** Re-save markdown file after finalize if `state.report_output_path` exists
- [ ] **3.5** Log finalize metadata to audit trail
- [ ] **3.6** Ensure `_flush_state` at line 552 captures the finalized report

## Phase 4: Add diagnostic safety net at export point

- [x] **4.1** In `action_handlers.py` `_get_report()`, import and call `needs_renumber` on `state.report`
- [x] **4.2** Log a warning if `needs_renumber` returns True (do not modify report)

## Phase 5: Tests

- [x] **5.1** Unit test `cleanup_citations`: strips LLM sources, removes dangling, does NOT renumber, does NOT append bibliography
- [x] **5.2** Unit test `finalize_citations`: renumbers to reading order, appends bibliography
- [x] **5.3** Unit test `needs_renumber`: True for `[3] then [1]`, False for `[1] then [2]`, False for no citations
- [x] **5.4** Integration test: `cleanup_citations` → simulate claim verification mutations (remove some `[N]`, remap others) → `finalize_citations` → verify sequential `[1,2,3,...]` output
- [x] **5.5** Backward compat test: `postprocess_citations` still produces identical output to pre-refactor behavior
- [x] **5.6** Verify existing `test_citation_tracking.py` and `test_citation_postprocess.py` tests pass without modification

## Phase 6: Validation

- [ ] **6.1** Run full test suite: `pytest tests/core/research/workflows/test_citation_tracking.py tests/core/research/workflows/deep_research/test_citation_postprocess.py -v`
- [ ] **6.2** Run broader test suite to check for regressions: `pytest tests/ -x --timeout=60`
- [ ] **6.3** Verify no import errors or circular dependencies from the new function exports
