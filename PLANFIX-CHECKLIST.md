# Post-Review Fix Plan v4 — Checklist

> Track implementation progress for [PLANFIX.md](PLANFIX.md).
> Mark items `[x]` as completed.

---

## FIX-0: Correctness & Data Integrity

### Item 0.1: Remove Duplicate Metadata Key in OpenAlex Provider
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [x] Remove duplicate `cited_by_count` key from metadata dict (kept `citation_count` — the canonical cross-provider key used by crossref, semantic_scholar, and all consumers)
- [x] Keep only `citation_count` key (canonical cross-provider convention)
- [x] Grep for consumers of `metadata["cited_by_count"]` — none found, removal is safe
- [x] No test updates needed — all tests use `citation_count`

#### Item 0.1 Validation

- [x] No duplicate keys in metadata dict
- [x] Downstream consumers (influence scoring, landscape, structured output) use correct key
- [x] Existing OpenAlex tests pass unchanged (114 tests + 134 downstream tests)

---

### Item 0.2: Fix Provider Fallback Naming in Topic Research
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] In `_handle_citation_search_tool()`: track last attempted provider name through fallback loop
- [x] Set `provider_name` to last attempted provider (e.g. `"openalex"`) instead of `"unknown"`
- [x] Apply same fix in `_handle_related_papers_tool()`
- [x] When no provider is available at all, use `"no_provider_available"`

#### Item 0.2 Validation

- [x] Response message names the actual provider when one succeeds
- [x] Response message names the last-attempted provider when all fail
- [x] No `"unknown"` provider name in any tool response

---

### Item 0.3: Simplify Redundant Boolean Check in Coverage Scoring
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py`

- [x] Change `if influential_count and influential_count > 0:` to `if influential_count > 0:` (line ~354)

#### Item 0.3 Validation

- [x] Existing supervision coverage tests pass unchanged

---

## FIX-1: Config Validation

### Item 1.1: Validate Citation Influence Threshold Ordering
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add validation in `__post_init__()` that `low < medium < high` for citation thresholds
- [x] Raise `ValueError` with descriptive message on mismatch
- [x] Add unit test: valid thresholds (5, 20, 100) pass without error
- [x] Add unit test: inverted thresholds (100, 20, 5) raise `ValueError`
- [x] Add unit test: equal thresholds (20, 20, 20) raise `ValueError`

#### Item 1.1 Validation

- [x] Default config passes validation
- [x] Custom valid thresholds pass
- [x] Inverted thresholds raise with clear error message
- [x] Existing config tests pass unchanged (34 tests in test_config_phase4.py)

---

### Item 1.2: Address Hardcoded Provider List in Brief
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [x] Add comment documenting that `_KNOWN_PROVIDERS` must stay in sync with provider registry
- [x] Move constant to top of file for visibility (extracted from inline default to module-level constant)
- [ ] ~~OR: query provider registry dynamically~~ (opted for option (b) — documented constant)

#### Item 1.2 Validation

- [x] Existing brief tests pass unchanged (84 brief-related tests)
- [x] Comment clearly explains sync requirement

---

## FIX-2: Test Coverage Gaps

### Item 2.1: Add Citation Tool Gating Tests
> **File**: `tests/core/research/workflows/test_topic_research.py`

- [x] Add `test_citation_search_rejected_when_gated()`:
  - [x] Create profile with `enable_citation_tools=False`
  - [x] Simulate researcher calling `citation_search` tool
  - [x] Verify rejection response with "not available" message
- [x] Add `test_related_papers_rejected_when_gated()`:
  - [x] Same setup with `enable_citation_tools=False`
  - [x] Verify rejection for `related_papers` tool
  - [x] Verify budget is NOT consumed on rejection

#### Item 2.1 Validation

- [x] Both tests pass
- [x] Tests fail if gating logic is removed (guard against regression)

---

### Item 2.2: Add Provider Fallback Chain Tests
> **File**: `tests/core/research/workflows/test_topic_research.py`

- [x] Add `test_citation_search_falls_back_to_openalex()`:
  - [x] Mock Semantic Scholar to raise exception
  - [x] Mock OpenAlex to return results
  - [x] Verify OpenAlex results are used
- [x] Add `test_related_papers_falls_back_to_openalex()`:
  - [x] Mock S2 `get_recommendations()` to raise
  - [x] Mock OpenAlex `get_related()` to return results
  - [x] Verify fallback works
- [x] Add `test_citation_search_both_providers_fail()`:
  - [x] Mock both providers to raise
  - [x] Verify empty result returned, no crash
  - [x] Verify provider_name is descriptive (per FIX-0.2)

#### Item 2.2 Validation

- [x] All three tests pass
- [x] Fallback order confirmed: Semantic Scholar first, OpenAlex second

---

### Item 2.3: Add OpenAlex `get_related()` and OOM Cap Tests
> **File**: `tests/core/research/providers/test_openalex.py`

- [x] Add `test_get_related()`:
  - [x] Mock OpenAlex response with `related_works` field
  - [x] Verify correct results returned
- [x] Add `test_get_related_no_related_works()`:
  - [x] Mock work with empty `related_works` list
  - [x] Verify empty list returned gracefully
- [x] Add `test_abstract_reconstruction_oom_cap()`:
  - [x] Create inverted index with position > 100,000
  - [x] Verify `_reconstruct_abstract()` returns `None`
  - [x] Added bonus `test_abstract_reconstruction_at_cap_succeeds()` — position exactly at cap still works

#### Item 2.3 Validation

- [x] All tests pass (4 tests: 2 get_related + 2 OOM cap)
- [x] OOM test confirms safety cap works

---

### Item 2.4: Add Paper ID Validation Tests
> **File**: `tests/core/research/workflows/test_topic_research.py`

- [x] Add `test_validate_paper_id_accepts_doi()` — `"10.1234/test.2024"` passes
- [x] Add `test_validate_paper_id_accepts_s2_hex()` — `"a1b2c3d4e5f6"` passes
- [x] Add `test_validate_paper_id_accepts_arxiv()` — `"2301.12345"` passes
- [x] Add `test_validate_paper_id_accepts_openalex()` — `"W2741809807"` passes
- [x] Add `test_validate_paper_id_rejects_injection()` — `"10.1234; DROP TABLE"` raises
- [x] Add `test_validate_paper_id_rejects_too_long()` — 257+ char string raises

#### Item 2.4 Validation

- [x] All six tests pass
- [x] Regex covers DOI, S2, ArXiv, OpenAlex, PubMed formats

---

### Item 2.5: Add OpenAlex 403 Error Test
> **File**: `tests/core/research/providers/test_openalex.py`

- [x] Add `test_403_raises_authentication_error()`:
  - [x] Mock 403 HTTP response
  - [x] Verify `AuthenticationError` is raised
  - [x] Verify error message includes provider name

#### Item 2.5 Validation

- [x] Test passes
- [x] Error type matches 401 behavior (both raise `AuthenticationError`)

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
