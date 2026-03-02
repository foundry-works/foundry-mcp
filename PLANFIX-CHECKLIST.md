# Post-Review Fix Plan v4 — Checklist

> Track implementation progress for [PLANFIX.md](PLANFIX.md).
> Mark items `[x]` as completed.

---

## FIX-0: Correctness & Data Integrity

### Item 0.1: Remove Duplicate Metadata Key in OpenAlex Provider
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [ ] Remove `citation_count` key from metadata dict (line ~616)
- [ ] Keep only `cited_by_count` key (matches OpenAlex API field name)
- [ ] Grep for consumers of `metadata["citation_count"]` and update to `cited_by_count`
- [ ] Update any test assertions referencing `citation_count` metadata key

#### Item 0.1 Validation

- [ ] No duplicate keys in metadata dict
- [ ] Downstream consumers (influence scoring, landscape, structured output) use correct key
- [ ] Existing OpenAlex tests pass unchanged (or updated)

---

### Item 0.2: Fix Provider Fallback Naming in Topic Research
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [ ] In `_handle_citation_search_tool()`: track last attempted provider name through fallback loop
- [ ] Set `provider_name` to last attempted provider (e.g. `"openalex"`) instead of `"unknown"`
- [ ] Apply same fix in `_handle_related_papers_tool()`
- [ ] When no provider is available at all, use `"no_provider_available"`

#### Item 0.2 Validation

- [ ] Response message names the actual provider when one succeeds
- [ ] Response message names the last-attempted provider when all fail
- [ ] No `"unknown"` provider name in any tool response

---

### Item 0.3: Simplify Redundant Boolean Check in Coverage Scoring
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py`

- [ ] Change `if influential_count and influential_count > 0:` to `if influential_count > 0:` (line ~354)

#### Item 0.3 Validation

- [ ] Existing supervision coverage tests pass unchanged

---

## FIX-1: Config Validation

### Item 1.1: Validate Citation Influence Threshold Ordering
> **File**: `src/foundry_mcp/config/research.py`

- [ ] Add validation in `__post_init__()` that `low < medium < high` for citation thresholds
- [ ] Raise `ValueError` with descriptive message on mismatch
- [ ] Add unit test: valid thresholds (5, 20, 100) pass without error
- [ ] Add unit test: inverted thresholds (100, 20, 5) raise `ValueError`
- [ ] Add unit test: equal thresholds (20, 20, 20) raise `ValueError`

#### Item 1.1 Validation

- [ ] Default config passes validation
- [ ] Custom valid thresholds pass
- [ ] Inverted thresholds raise with clear error message
- [ ] Existing config tests pass unchanged

---

### Item 1.2: Address Hardcoded Provider List in Brief
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [ ] Add comment documenting that `_KNOWN_PROVIDERS` must stay in sync with provider registry
- [ ] Move constant to top of file for visibility (if not already)
- [ ] OR: query provider registry dynamically (if low complexity)

#### Item 1.2 Validation

- [ ] Existing brief tests pass unchanged
- [ ] Comment clearly explains sync requirement

---

## FIX-2: Test Coverage Gaps

### Item 2.1: Add Citation Tool Gating Tests
> **File**: `tests/core/research/workflows/deep_research/` or `tests/core/research/workflows/test_topic_research.py`

- [ ] Add `test_citation_search_rejected_when_gated()`:
  - [ ] Create profile with `enable_citation_tools=False`
  - [ ] Simulate researcher calling `citation_search` tool
  - [ ] Verify rejection response with "not available" message
- [ ] Add `test_related_papers_rejected_when_gated()`:
  - [ ] Same setup with `enable_citation_tools=False`
  - [ ] Verify rejection for `related_papers` tool
  - [ ] Verify budget is NOT consumed on rejection

#### Item 2.1 Validation

- [ ] Both tests pass
- [ ] Tests fail if gating logic is removed (guard against regression)

---

### Item 2.2: Add Provider Fallback Chain Tests
> **File**: `tests/core/research/workflows/deep_research/` or `tests/core/research/workflows/test_topic_research.py`

- [ ] Add `test_citation_search_falls_back_to_openalex()`:
  - [ ] Mock Semantic Scholar to raise exception
  - [ ] Mock OpenAlex to return results
  - [ ] Verify OpenAlex results are used
- [ ] Add `test_related_papers_falls_back_to_openalex()`:
  - [ ] Mock S2 `get_recommendations()` to raise
  - [ ] Mock OpenAlex `get_related()` to return results
  - [ ] Verify fallback works
- [ ] Add `test_citation_search_both_providers_fail()`:
  - [ ] Mock both providers to raise
  - [ ] Verify empty result returned, no crash
  - [ ] Verify provider_name is descriptive (per FIX-0.2)

#### Item 2.2 Validation

- [ ] All three tests pass
- [ ] Fallback order confirmed: Semantic Scholar first, OpenAlex second

---

### Item 2.3: Add OpenAlex `get_related()` and OOM Cap Tests
> **File**: `tests/core/research/providers/test_openalex.py`

- [ ] Add `test_get_related()`:
  - [ ] Mock OpenAlex response with `related_works` field
  - [ ] Verify correct results returned
- [ ] Add `test_get_related_no_related_works()`:
  - [ ] Mock work with empty `related_works` list
  - [ ] Verify empty list returned gracefully
- [ ] Add `test_abstract_reconstruction_oom_cap()`:
  - [ ] Create inverted index with position > 100,000
  - [ ] Verify `_reconstruct_abstract()` returns `None`
  - [ ] Verify warning is logged

#### Item 2.3 Validation

- [ ] All three tests pass
- [ ] OOM test confirms safety cap works

---

### Item 2.4: Add Paper ID Validation Tests
> **File**: `tests/core/research/workflows/test_topic_research.py`

- [ ] Add `test_validate_paper_id_accepts_doi()` — `"10.1234/test.2024"` passes
- [ ] Add `test_validate_paper_id_accepts_s2_hex()` — `"a1b2c3d4e5f6"` passes
- [ ] Add `test_validate_paper_id_accepts_arxiv()` — `"2301.12345"` passes
- [ ] Add `test_validate_paper_id_accepts_openalex()` — `"W2741809807"` passes
- [ ] Add `test_validate_paper_id_rejects_injection()` — `"10.1234; DROP TABLE"` raises
- [ ] Add `test_validate_paper_id_rejects_too_long()` — 257+ char string raises

#### Item 2.4 Validation

- [ ] All six tests pass
- [ ] Regex covers DOI, S2, ArXiv, OpenAlex, PubMed formats

---

### Item 2.5: Add OpenAlex 403 Error Test
> **File**: `tests/core/research/providers/test_openalex.py`

- [ ] Add `test_403_raises_authentication_error()`:
  - [ ] Mock 403 HTTP response
  - [ ] Verify `AuthenticationError` is raised
  - [ ] Verify error message includes provider name

#### Item 2.5 Validation

- [ ] Test passes
- [ ] Error type matches 401 behavior (both raise `AuthenticationError`)

---

## FIX-3: Cleanup & Consistency

### Item 3.1: Update PLAN-CHECKLIST.md Completion Status
> **File**: `PLAN-CHECKLIST.md`

- [ ] Audit Phase 0 items against codebase — mark completed items `[x]`
  - [ ] Item 0.1a: supervision_first_round.py exists with extracted functions
  - [ ] Item 0.1b: Evaluate helper extraction (document decision)
  - [ ] Item 0.1c: Thin wrapper methods (check current line count)
  - [ ] Item 0.2: ResearchExtensions container implemented
- [ ] Audit Phase 1 items — mark completed items `[x]`
- [ ] Audit Phase 2 items — mark completed items `[x]`
- [ ] Audit Phase 3 items — mark completed items `[x]`
- [ ] Audit Phase 4 items — mark completed items `[x]`
- [ ] Mark Final Validation items as appropriate

#### Item 3.1 Validation

- [ ] PLAN-CHECKLIST.md accurately reflects implementation state
- [ ] No items marked complete that aren't actually implemented
- [ ] No items left unchecked that are implemented in code

---

### Item 3.2: Remove Orphan Research Output Files from Repo Root

- [ ] Delete or `.gitignore` these untracked files:
  - [ ] `compare-postgresql-vs-mysql-for-oltp-workloads-in-2024*.md` (3 files)
  - [ ] `test-progress-heartbeat*.md` (2 files)
  - [ ] `test-synthesis-heartbeat*.md` (2 files)
  - [ ] `what-are-the-benefits-of-renewable-energy*.md` (2 files)
- [ ] Add `*.md` pattern to root `.gitignore` for deep research outputs, OR add specific patterns

#### Item 3.2 Validation

- [ ] `git status` no longer shows these as untracked
- [ ] Legitimate markdown files (README.md, PLAN.md, CHANGELOG.md, etc.) are NOT ignored

---

### Item 3.3: Document StructuredResearchOutput Dict Schemas
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [ ] Add field descriptions to `StructuredResearchOutput.sources` documenting expected keys
- [ ] Add field descriptions to `StructuredResearchOutput.findings` documenting expected keys
- [ ] Add field descriptions to `StructuredResearchOutput.gaps` documenting expected keys
- [ ] Add field descriptions to `StructuredResearchOutput.contradictions` documenting expected keys

#### Item 3.3 Validation

- [ ] Each list[dict] field has documented key schema in Field description
- [ ] Documentation matches actual keys produced by `_build_structured_output()`

---

### Item 3.4: Add `generated_at` Timestamps to Output Models
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [ ] Add `generated_at: Optional[str] = None` to `StructuredResearchOutput`
- [ ] Add `generated_at: Optional[str] = None` to `CitationNetwork`
- [ ] Populate `generated_at` in `_build_structured_output()` with ISO 8601 timestamp
- [ ] Populate `generated_at` in `CitationNetworkBuilder.build_network()` with ISO 8601 timestamp

#### Item 3.4 Validation

- [ ] `StructuredResearchOutput.generated_at` is populated after synthesis
- [ ] `CitationNetwork.generated_at` is populated after network build
- [ ] Timestamp is valid ISO 8601 format
- [ ] Existing serialization tests pass (field is Optional, backward compatible)

---

## Final Validation

- [ ] All 7,607+ existing tests still pass
- [ ] No new test failures introduced
- [ ] New tests (FIX-2) all pass
- [ ] `git diff --stat` shows only expected changes
- [ ] No security regressions (SSRF, injection protection intact)

---

## Estimated Scope

| Fix | Impl LOC | Test LOC |
|-----|----------|----------|
| 0. Correctness & data integrity | ~30-50 | ~0 |
| 1. Config validation | ~15-25 | ~20-30 |
| 2. Test coverage gaps | ~0 | ~150-250 |
| 3. Cleanup & consistency | ~20-40 | ~0 |
| **Total** | **~65-115** | **~170-280** |
